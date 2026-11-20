/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tf_adapter/kernels/geop_npu.h"

#include <chrono>
#include <cstdint>
#include <dirent.h>
#include <dlfcn.h>
#include <fstream>
#include <sstream>
#include <map>
#include <memory>
#include <mmpa/mmpa_api.h>
#include <queue>
#include <securec.h>
#include <securectype.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>

#include "tf_adapter/common/adapter_logger.h"
#include "tf_adapter/common/common.h"
#include "tf_adapter/util/ge_plugin.h"
#include "tf_adapter/util/infershape_util.h"
#include "tf_adapter/util/npu_attrs.h"
#include "tf_adapter/util/generate_report.h"
#include "tf_adapter/util/npu_ops_identifier.h"
#include "tf_adapter/util/session_manager.h"

#ifdef TF_VERSION_TF2
#include "tensorflow/compiler/tf2xla/functionalize_control_flow_util.h"
#endif
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/env_var.h"
#include "external/aoe.h"
#include "external/aoe_errcodes.h"

#include "common/ge_common/scope_guard.h"
#include "parser/onnx_parser.h"
#include "ge/ge_api.h"
#include "ge_common/ge_api_types.h"
#include "graph/ascend_string.h"
#include "tf_adapter_2.x/npu_device/core/npu_micros.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tf_adapter/util/profiler.h"
#include "ge/ge_api_wrapper.h"
#include "register/register_types.h"
#include "graph/def_types.h"
#include "tf_adapter/util/scoped_graph_manager_interface.h"
#include "tf_adapter/util/scoped_graph_manager.h"

namespace tensorflow {
#ifdef TF_VERSION_TF2
Status FunctionalizeControlFlow(Graph *graph, FunctionLibraryDefinition *library, const NodeFilter &node_filter = {},
                                bool include_functions = false);
#else
Status FunctionalizeControlFlow(Graph *graph, FunctionLibraryDefinition *library);
#endif

void MarkDataNodesAsHostTensor(ge::Graph &graph) {
  for (ge::GNode &node : graph.GetDirectNode()) {
    ge::AscendString node_type;
    if (node.GetType(node_type) != ge::GRAPH_SUCCESS) {
      continue;
    }
    const char *type = node_type.GetString();
    if ((type != nullptr) && (std::strcmp(type, "Data") == 0)) {
      ge::AscendString attr_name("_host_tensor");
      bool is_host_tensor = true;
      (void)node.SetAttr(attr_name, is_host_tensor);
    }
  }
}

namespace {
const std::string ATTR_NAME_CONST_INPUT_NAME = "_const_input";
const std::string kAutoRecompute = "auto";
const std::string kTotalStep = "TOTAL_STEP";
const std::string kStepNow = "STEP_NOW";
const std::string kTargetLoss = "TARGET_LOSS";
const std::string kLossNow = "LOSS_NOW";
const std::string kModeValueStep = "step";
const std::string kModeValueLoss = "loss";
const float kDefaultStepRatio = 0.9;
const float kMinStepRatio = 0.2;
const float kMaxStepRatio = 0.9;
const float kDefaultLossRatio = 1.05;
const float kMinLossRatio = 1.01;
const float kMaxLossRatio = 1.5;
const std::map<std::string, GeOp::FastValue> fast_value_string_2_eunm = {{"fast", GeOp::FastValue::kfast},
                                                                         {"fast1", GeOp::FastValue::kfast1}};

const std::map<GeOp::FastValue, std::string> fast_value_enum_2_string = {{GeOp::FastValue::kfast, "fast"},
                                                                         {GeOp::FastValue::kfast1, "fast1"}};
const std::map<GeOp::FastValue, std::string> fast_value_2_precision_mode_v1 = {
    {GeOp::FastValue::kfast, "allow_mix_precision_fp16"},
    {GeOp::FastValue::kfast1, "allow_mix_precision_bf16"},
};
const std::unordered_set<std::string> supported_origin_precision_mode_v1 = {"allow_fp32_to_fp16",
                                                                            "must_keep_origin_dtype", ""};
const std::unordered_set<std::string> valid_mode_values = {kModeValueStep, kModeValueLoss};
const std::map<GeOp::FastValue, std::string> fast_value_2_precision_mode_v2 = {
    {GeOp::FastValue::kfast, "mixed_float16"}, {GeOp::FastValue::kfast1, "mixed_bfloat16"}};
const std::unordered_set<std::string> supported_origin_precision_mode_v2 = {"origin", ""};
using geDataUniquePtr = std::unique_ptr<uint8_t[], std::function<void(uint8_t *)>>;

class NpuHostFixedAllocator : public tensorflow::Allocator, public tensorflow::core::RefCounted {
 public:
  static tensorflow::Allocator *Create(geDataUniquePtr ptr) {
    return new (std::nothrow) NpuHostFixedAllocator(std::move(ptr));
  }

 private:
  explicit NpuHostFixedAllocator(geDataUniquePtr ptr) : ptr_(std::move(ptr)) {}
  ~NpuHostFixedAllocator() override {}
  std::string Name() override {
    return "NpuHostFixedAllocator";
  }
  void *AllocateRaw(size_t alignment, size_t num_bytes) override {
    (void)alignment;
    (void)num_bytes;
    return ptr_.get();
  }
  void DeallocateRaw(void *ptr) override {
    (void)ptr;
    Unref();
  }
  geDataUniquePtr ptr_;
};

class NpuGetNextOutputInfo {
 public:
  NpuGetNextOutputInfo(ge::Placement placement, std::vector<int64_t> &dims, size_t output_size, geDataUniquePtr data)
      : placement_(placement), dims_(dims), output_size_(output_size), data_(std::move(data)) {}
  ~NpuGetNextOutputInfo() {}
  ge::Placement placement_;
  std::vector<int64_t> dims_;
  size_t output_size_;
  geDataUniquePtr data_;
};

class NpuHostGetNextAllocator : public tensorflow::Allocator, public tensorflow::core::RefCounted {
 public:
  static tensorflow::Allocator *Create(std::unique_ptr<NpuGetNextOutputInfo> output) {
    return new (std::nothrow) NpuHostGetNextAllocator(std::move(output));
  }

 private:
  explicit NpuHostGetNextAllocator(std::unique_ptr<NpuGetNextOutputInfo> output) : output_(std::move(output)) {}
  ~NpuHostGetNextAllocator() override {}
  std::string Name() override {
    return "NpuHostGetNextAllocator";
  }
  void *AllocateRaw(size_t alignment, size_t num_bytes) override {
    (void)alignment;
    (void)num_bytes;
    return output_.get();
  }
  void DeallocateRaw(void *ptr) override {
    (void)ptr;
    Unref();
  }
  std::unique_ptr<NpuGetNextOutputInfo> output_;
};

inline string ToString(ge::Status status) {
  return ::ge::StatusFactory::Instance()->GetErrDescV2(status).GetString();
}

Status BuildStringOutput(geDataUniquePtr data_ptr, size_t output_size, Tensor &cpu_tensor) {
  TensorShape out_shape = cpu_tensor.shape();
  if ((out_shape.num_elements() * sizeof(ge::StringHead)) >= output_size) {
    LOG(ERROR) << "[GEOP] Graph engine process success but output string format is not right";
    return errors::Internal("Graph engine process graph success but output string format is not right.");
  }
  auto tensor_flat = cpu_tensor.flat<tstring>();
  tstring *tensor_data = tensor_flat.data();
  ge::StringHead *string_head = reinterpret_cast<ge::StringHead *>(reinterpret_cast<char *>(data_ptr.get()));
  for (int64_t j = 0; j < out_shape.num_elements(); j++) {
    int64_t offset = string_head[j].addr;
    int64_t string_len = string_head[j].len;
    const char *temp_string = reinterpret_cast<const char *>(data_ptr.get()) + offset;
    tensor_data[j] = tstring(temp_string, string_len);
    ADP_LOG(INFO) << "[GEOP] output string data " << tensor_data[j];
  }
  return Status::OK();
}

Status BuildOutputTensorInfo(OpKernelContext *ctx, std::vector<ge::Tensor> &outputs) {
  // ctx is not nullptr
  int num_outputs = ctx->num_outputs();
  ADP_LOG(INFO) << "BuildOutputTensorInfo, num_outputs:" << num_outputs;
  if (num_outputs != static_cast<int>(outputs.size())) {
    ADP_LOG(ERROR) << "[GEOP] Outputs num mismatched, need:" << num_outputs << ", while GE return:" << outputs.size();
    return errors::InvalidArgument("Outputs num mismatched, need:", num_outputs, ", while GE return:", outputs.size());
  }

  // populate outputs
  for (int i = 0; i < num_outputs; ++i) {
    ge::Tensor &output = outputs[i];
    std::vector<int64_t> ge_output_dims = output.GetTensorDesc().GetShape().GetDims();
    ge::Placement data_placement = output.GetTensorDesc().GetPlacement();
    std::vector<int64> dims;
    std::transform(ge_output_dims.begin(), ge_output_dims.end(), std::back_inserter(dims),
                   [](const int64_t dim) { return dim; });
    TensorShape out_shape(dims);
    const DataType out_type = ctx->op_kernel().output_type(i);
    size_t output_size = output.GetSize();
    geDataUniquePtr data_ptr = output.ResetData();

    ADP_LOG(INFO) << "[GEOP] Get ge output: " << i << " tensor shape is: " << out_shape.DebugString()
                  << ", data placement is: " << data_placement << ", output_size is: " << output_size
                  << ", data addr is: " << std::hex << reinterpret_cast<uintptr_t>(data_ptr.get());

    if (data_placement != ge::kPlacementDevice) {
      const static int64_t kTensorAlignBytes = 64;
      if (reinterpret_cast<uintptr_t>(data_ptr.get()) % kTensorAlignBytes == 0) {
        ADP_LOG(INFO) << "[GEOP] Zero copy ge tensor " << reinterpret_cast<uintptr_t>(data_ptr.get())
                      << " as aligned with " << kTensorAlignBytes << " bytes";

        if (out_type == DT_STRING) {  // string type op is not sink now
          Tensor cpu_tensor = Tensor(out_type, out_shape);
          if (BuildStringOutput(std::move(data_ptr), output_size, cpu_tensor) != Status::OK()) {
            return errors::Internal("The output string data analyze failed.");
          }
          ctx->set_output(i, cpu_tensor);
          continue;
        }
        if (out_shape.num_elements() != 0) {
          Allocator *allocator = NpuHostFixedAllocator::Create(std::move(data_ptr));
          Tensor cpu_tensor(allocator, out_type, out_shape);
          if (output_size != cpu_tensor.TotalBytes()) {
            LOG(ERROR) << "[GEOP] Graph engine process graph success but output " << i << " total bytes " << output_size
                       << " mismatched with expected " << cpu_tensor.TotalBytes();
            return errors::Internal("Graph engine process graph success but output length mismatched with expected.");
          }
          ctx->set_output(i, cpu_tensor);
          continue;
        }
        ctx->set_output(i, Tensor(out_type, out_shape));
      } else {
        ADP_LOG(ERROR) << "[GEOP] Skip zero copy as ge tensor, " << reinterpret_cast<uintptr_t>(data_ptr.get())
                       << " not aligned with " << kTensorAlignBytes << " bytes";
        return errors::Internal("[GEOP] Skip zero copy ge tensor, bytes not aligned with expected.");
      }
    } else {
      ADP_LOG(INFO) << "[GEOP] GE output data placement is device, construct output info tensor.";
      auto getnext_output_info = std::unique_ptr<NpuGetNextOutputInfo>(
          new NpuGetNextOutputInfo(data_placement, ge_output_dims, output_size, std::move(data_ptr)));
      Allocator *allocator = NpuHostGetNextAllocator::Create(std::move(getnext_output_info));
      Tensor cpu_tensor(allocator, out_type, out_shape);
      ctx->set_output(i, cpu_tensor);
    }
  }
  ADP_LOG(INFO) << "[GEOP] Build output tensor info success.";
  return Status::OK();
}

bool CmpValue(const std::pair<std::vector<string>, uint32_t> &p1, const std::pair<std::vector<string>, uint32_t> &p2) {
  return p1.second < p2.second;
}

bool CmpVecValue(const Node *const node1, const Node *const node2) {
  if (node1 == nullptr || node2 == nullptr) {
    ADP_LOG(ERROR) << "node1 or node2 is nullptr.";
    LOG(ERROR) << "node1 or node2 is nullptr.";
    return false;
  }
  return node1->name() < node2->name();
}

bool CmpNodeIndex(const std::pair<Node *, uint32_t> &p1, const std::pair<Node *, uint32_t> &p2) {
  return p1.second < p2.second;
}

void SetReuseOptions(const std::string &key, int32_t num, const std::map<std::string, std::string> &global_options,
                     const std::map<std::string, std::string> &init_options,
                     std::map<std::string, std::string> &options) {
  if (num < 1) {
    return;
  }
  auto inserted_kv = options.insert(std::make_pair(key, ""));
  if (inserted_kv.second) {
    for (int32_t i = 0; i < (num - 1); i++) {
      inserted_kv.first->second.append(std::to_string(i));
      inserted_kv.first->second.append(",");
    }
    inserted_kv.first->second.append(std::to_string(num - 1));
    ADP_LOG(INFO) << "Set reuse options, key: " << key << ", value: " << inserted_kv.first->second;
  }
}
class ExitCallbackGuarder {
 public:
  explicit ExitCallbackGuarder(std::function<void()> done) : done_(done) {}
  ~ExitCallbackGuarder() {
    done_();
  }

 private:
  std::function<void()> done_;
};

}  // namespace

std::string CurrentTimeInStr() {
  std::time_t now = std::time(nullptr);
  std::tm *ptm = std::localtime(&now);
  if (ptm == nullptr) {
    return "";
  }

  const int32_t time_buffer_len = 32;
  char buffer[time_buffer_len] = {0};
  // format: 20171122042550
  std::strftime(buffer, time_buffer_len, "%Y%m%d%H%M%S", ptm);
  return std::string(buffer);
}

void ReplaceTargetStr(std::string &str, const std::string &from, const std::string &to) {
  size_t pos = 0U;
  while ((pos = str.find(from, pos)) != std::string::npos) {
    str.replace(pos, from.length(), to);
    pos += to.length();
  }
}

void RewriteInputShapeOption(std::string &str) {
  // in case of a:-1,2,3;b:0
  str += ";";
  ReplaceTargetStr(str, ":0;", ":;");
  str.pop_back();
}

static const int64 kMicrosToMillis = 1000;
const int kInvalidGraphId = 0;
const int kMaxCacheNum = 10;
const int kFatalSleepTime = 3000;
const std::string kAllReduce = "HcomAllReduce";

GeOp::GeOp(OpKernelConstruction *ctx)
    : AsyncOpKernel(ctx),
      init_flag_(false),
      sess_init_flag_(false),
      graph_id_init_flag_(false),
      is_input_convert_(false),
      data_format_(""),
      graph_id_(0),
      is_initialized_graph_(false),
      is_empty_graph_(false),
      need_iteration_(false),
      tf_session_(""),
      ge_session_(nullptr),
      job_type_(""),
      is_host_graph_(false),
      handle_(nullptr),
      tuned_flag_(ATOMIC_FLAG_INIT),
      jit_compile_("2"),
      is_dynamic_input_(false),
      session_id_(0),
      aoe_initialize_(nullptr),
      aoe_finalize_(nullptr),
      aoe_create_session_(nullptr),
      aoe_destroy_session_(nullptr),
      aoe_set_gesession_(nullptr),
      aoe_set_dependgraphs_(nullptr),
      aoe_set_tuninggraph_(nullptr),
      aoe_tuning_graph_(nullptr),
      aoe_set_depend_graphs_inputs_(nullptr),
      aoe_set_tuning_graph_input_(nullptr),
      need_compile_graph_first_(false) {
  Initialize(ctx);
}

GeOp::~GeOp() {
  Finalize();
}

void GeOp::Initialize(OpKernelConstruction *ctx) {
  mutex_lock lock{mu_};
  int64 startTime = InferShapeUtil::GetCurrentTimestap();
  ADP_LOG(INFO) << "[GEOP] Begin to GeOp initialize.";
  if (init_flag_) {
    ADP_LOG(WARNING) << "[GEOP] GEOP already Initialize.";
    return;
  }

  CHECK_NOT_NULL(ctx);
  const NameAttrList *func = nullptr;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("function", &func));
  function_ = *func;
  std::string data_format;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
  ADP_LOG(INFO) << "Attr 'data_format' of " << ctx->def().name() << " is " << data_format;
  this->data_format_ = data_format;

  Status s = ctx->GetAttr("_session", &tf_session_);
  if (s.ok()) {
    ADP_LOG(INFO) << "[GEOP] get session info from attr, tf session: " << tf_session_;
  }

  (void)ctx->GetAttr("_recompute_mode", &recompute_mode_);
  (void)ctx->GetAttr("_compile_dynamic_mode", &compile_dynamic_mode_);
  (void)ctx->GetAttr("_dynamic_input", &dynamic_input_);
  (void)ctx->GetAttr("_jit_compile", &jit_compile_);
  if (!dynamic_input_.empty() && dynamic_input_ == "1") {
    jit_compile_ = "1";
    is_dynamic_input_ = true;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("_dynamic_graph_execute_mode", &dynamic_graph_execute_mode_));
    (void)ctx->GetAttr("_getnext_inputs_shape_range", &getnext_inputs_shape_range_);
    (void)ctx->GetAttr("_data_inputs_shape_range", &data_inputs_shape_range_);
    (void)ctx->GetAttr("_is_dynamic_getnext", &is_dynamic_getnext_);
    (void)ctx->GetAttr("_placeholder_index", &placeholder_index_);
  }
  (void)ctx->GetAttr("_train_graph", &is_train_graph_);
  (void)ctx->GetAttr("_is_var_init_graph", &is_var_init_graph_);
  (void)ctx->GetAttr("_shape_generalization_mode", &shape_generalization_mode_);
  ADP_LOG(INFO) << "[GEOP] dynamic_input: " << dynamic_input_
                << ", dynamic_graph_execute_mode: " << dynamic_graph_execute_mode_ << ", jit_compile: " << jit_compile_
                << ", is_dynamic_input: " << is_dynamic_input_
                << ", getnext_inputs_shape_range: " << getnext_inputs_shape_range_
                << ", data_inputs_shape_range: " << data_inputs_shape_range_ << ", is_train_graph: " << is_train_graph_
                << ", is_dynamic_getnext: " << is_dynamic_getnext_ << ", placeholder_index: " << placeholder_index_
                << ", is_var_init_graph: " << is_var_init_graph_ << ", compile_dynamic_mode: " << compile_dynamic_mode_
                << ", shape_generalization_mode: " << shape_generalization_mode_;

  if (compile_dynamic_mode_ == "1" && shape_generalization_mode_ != "STRICT") {
    ADP_LOG(WARNING) << "compile_dynamic_mode is set to true, shape_generalization_mode[" << shape_generalization_mode_
                     << "] will be ignore, please set compile_dynamic_mode=false.";
  }
  if (jit_compile_ != "1" && shape_generalization_mode_ != "STRICT") {
    LOG(WARNING) << "jit_compile is not set to true, shape_generalization_mode[" << shape_generalization_mode_
                 << "] will be ignore, please set jit_compile=true "
                 << "and shape_generalization_mode=" << shape_generalization_mode_ << ".";
    ADP_LOG(WARNING) << "jit_compile is not set to true, shape_generalization_mode[" << shape_generalization_mode_
                     << "] will be ignore, please set jit_compile=true "
                     << "and shape_generalization_mode=" << shape_generalization_mode_ << ".";
  }
  // global environment Initialize, invoke once for each process
  std::string sess_config = "";
  OP_REQUIRES_OK(ctx, ctx->GetAttr("_NpuOptimizer", &sess_config));
  std::map<std::string, std::string> pass_options = NpuAttrs::GetPassOptions(ctx);
  iteration_per_loop_ = std::atoi(pass_options["iterations_per_loop"].c_str());
  graph_max_parallel_model_num_ = std::max(std::atoi(pass_options["graph_max_parallel_model_num"].c_str()), 1);
  ADP_LOG(INFO) << "graph_max_parallel_model_num: " << graph_max_parallel_model_num_;
  job_type_ = pass_options["job"];
  mix_compile_mode_ = pass_options["mix_compile_mode"];
  accelerate_train_mode_ = pass_options["accelerate_train_mode"];
  ADP_LOG(INFO) << "accelerate train mode: " << accelerate_train_mode_;
  if (GePlugin::GetInstance()->IsGlobal()) {
    ADP_LOG(INFO) << "[GEOP] GePlugin global, skip GePlugin init";
    InitAoeFlag();
  } else {
    init_options_ = NpuAttrs::GetInitOptions(ctx);
    InitAoeFlag();
    // aoe should not init ge async
    GePlugin::GetInstance()->Init(init_options_, false, !is_aoe_);
    ADP_LOG(INFO) << "[GEOP] GePlugin initialized successfully.";
  }
  ADP_LOG(INFO) << "init options: ";
  if (is_aoe_) {
    handle_ = mmDlopen("libaoe_tuning.so", MMPA_RTLD_NOW);
    OP_REQUIRES(ctx, handle_ != nullptr, errors::InvalidArgument("libaoe_tuning.so dlopen failed, ", mmDlerror()));
    // aoe init
    aoe_initialize_ = (AoeInitializeFunc)mmDlsym(handle_, "AoeInitialize");
    OP_REQUIRES(ctx, aoe_initialize_ != nullptr,
                errors::InvalidArgument("dlsym Aoe initialize API failed, ", mmDlerror()));
    // aoe finalize
    aoe_finalize_ = (AoeFinalizeFunc)mmDlsym(handle_, "AoeFinalize");
    OP_REQUIRES(ctx, aoe_initialize_ != nullptr,
                errors::InvalidArgument("dlsym Aoe Finalize API failed, ", mmDlerror()));
    // aoe create session
    aoe_create_session_ = (AoeCreateSessionFunc)mmDlsym(handle_, "AoeCreateSession");
    OP_REQUIRES(ctx, aoe_create_session_ != nullptr,
                errors::InvalidArgument("dlsym Aoe create session API failed, ", mmDlerror()));
    // aoe destroy session
    aoe_destroy_session_ = (AoeDestroySessionFunc)mmDlsym(handle_, "AoeDestroySession");
    OP_REQUIRES(ctx, aoe_destroy_session_ != nullptr,
                errors::InvalidArgument("dlsym Aoe destroy session API failed, ", mmDlerror()));
    // share ge_session to aoe
    aoe_set_gesession_ = (AoeSetGeSessionFunc)mmDlsym(handle_, "AoeSetGeSession");
    OP_REQUIRES(ctx, aoe_set_gesession_ != nullptr,
                errors::InvalidArgument("dlsym Aoe set session API failed, ", mmDlerror()));
    // aoe set depend graphs
    aoe_set_dependgraphs_ = (AoeSetDependGraphFunc)mmDlsym(handle_, "AoeSetDependGraphs");
    OP_REQUIRES(ctx, aoe_set_dependgraphs_ != nullptr,
                errors::InvalidArgument("dlsym Aoe set depend graphs API failed, ", mmDlerror()));
    // aoe set tuning graph
    aoe_set_tuninggraph_ = (AoeSetTuningGraphFunc)mmDlsym(handle_, "AoeSetTuningGraph");
    OP_REQUIRES(ctx, aoe_set_tuninggraph_ != nullptr,
                errors::InvalidArgument("dlsym Aoe aoe set tuning graph API failed, ", mmDlerror()));
    // aoe tuning
    aoe_tuning_graph_ = (AoeTuningGraphFunc)mmDlsym(handle_, "AoeTuningGraph");
    OP_REQUIRES(ctx, aoe_tuning_graph_ != nullptr,
                errors::InvalidArgument("dlsym Aoe tuning graph API failed, ", mmDlerror()));
    // aoe set tuning depend graphs inputs
    aoe_set_depend_graphs_inputs_ =
        reinterpret_cast<AoeSetDependGraphsInputsFunc>(mmDlsym(handle_, "AoeSetDependGraphsInputs"));
    OP_REQUIRES(ctx, aoe_set_depend_graphs_inputs_ != nullptr,
                errors::InvalidArgument("dlsym Aoe set tuning depend graphs inputs API failed, ", mmDlerror()));
    // aoe set tuning graph inputs
    aoe_set_tuning_graph_input_ =
        reinterpret_cast<AoeSetTuningGraphInputFunc>(mmDlsym(handle_, "AoeSetTuningGraphInput"));
    OP_REQUIRES(ctx, aoe_set_tuning_graph_input_ != nullptr,
                errors::InvalidArgument("dlsym Aoe set tuning graph inputs API failed, ", mmDlerror()));
  }

  sess_options_ = NpuAttrs::GetSessOptions(ctx);
  graph_options_ = NpuAttrs::GetGraphOptions(ctx);
  input_shapes_vec_.resize(ctx->num_inputs() + 1, absl::nullopt);

  init_flag_ = true;
  int64 endTime = InferShapeUtil::GetCurrentTimestap();
  ADP_LOG(EVENT) << "[GEOP] GeOp initialized successfully, cost:[" << ((endTime - startTime) / kMicrosToMillis)
                 << " ms].";
  return;
}

void GeOp::Finalize() {
  {
    ADP_LOG(INFO) << "[GEOP] GeOp starts to finalize, tf session: " << tf_session_ << ", graph_id_: " << graph_id_;
    // global environment finalize, invoke once for each process
    {
      mutex_lock lock{mu_};
      uint32_t graph_id = -1;
      if ((sess_init_flag_ && graph_id_init_flag_) || !tf_session_.empty()) {
        bool ret = DecrementGraphIdCount(graph_id);
        if (!ret) {
          ADP_LOG(ERROR) << "tf session " << tf_session_ << " sub graph id failed.";
          LOG(ERROR) << "tf session " << tf_session_ << " sub graph id failed.";
          return;
        }
        graph_id_init_flag_ = false;
        if (graph_id == kInvalidGraphId) {
          SessionManager::GetInstance().DestroyGeSession(tf_session_);
          ClearGraphIdCount();
          sess_init_flag_ = false;
        }
      }

      if (!SessionManager::GetInstance().IsGeSessionExist()) {
        if (!GePlugin::GetInstance()->IsGlobal()) {
          GePlugin::GetInstance()->Finalize();
          ADP_LOG(INFO) << "[GEOP] GePlugin finalized successfully.";
          if (!init_options_["ge.jobType"].empty() && !init_options_["ge.tuningPath"].empty() &&
              aoe_finalize_ != nullptr && tuned_initialize_flag_) {
            AoeStatus tune_ret = (*aoe_finalize_)();
            if (tune_ret != Aoe::AOE_SUCCESS) {
              ADP_LOG(ERROR) << "[GEOP] exec aoe finalize func failed.";
              LOG(ERROR) << "[GEOP] exec aoe finalize func failed.";
              return;
            }
          }
          tuned_initialize_flag_ = false;
        } else {
          ADP_LOG(INFO) << "[GEOP] GePlugin global, skip GePlugin Finalize";
        }
        if (!GenerateReport::GetInstance()->SaveUnsupportedInfo().ok()) {
          ADP_LOG(WARNING) << "[GEOP] Save check report failed.";
          LOG(WARNING) << "[GEOP] Save check report failed.";
        }
        if (handle_ != nullptr) {
          (void)mmDlclose(handle_);
        }
      }
    }
  }
  init_flag_ = false;
  ADP_LOG(INFO) << "[GEOP] GeOp finalized successfully, tf session: " << tf_session_ << ", graph_id_: " << graph_id_;
  return;
}

uint32_t GetStepToChange(const uint32_t total_step, const float ratio) {
  return total_step * ratio;
}

float GetLossToChange(const float target_loss, const float ratio) {
  return target_loss * ratio;
}

Status GeOp::AccelerateInfo::TriggeredByStep(bool &is_triggered) {
  uint32_t total_step = 0U;
  REQUIRES_STATUS_OK(GetStepFromEnv(kTotalStep, total_step));
  uint32_t step_to_change = GetStepToChange(total_step, fast_ratio_);
  uint32_t step_now = 0U;
  REQUIRES_STATUS_OK(GetStepFromEnv(kStepNow, step_now));
  ADP_LOG(INFO) << "[GEOP] accelerate train: get expected trigger recompile step: " << step_to_change
                << " with total step: " << total_step << " and step now is: " << step_now;
  if ((step_now >= step_to_change) || (step_now + iteration_per_loop_ >= total_step)) {
    ADP_LOG(EVENT) << "[GEOP] accelerate train: trigger recompile when step is " << step_now;
    if (step_now != step_to_change) {
      ADP_LOG(WARNING) << "[GEOP] accelerate train: trigger recompile step earlier or later than expected step, may"
                          " have some effect on train";
    }
    is_triggered = true;
    is_recovered_ = true;
    return Status::OK();
  }
  is_triggered = false;
  return Status::OK();
}

Status GeOp::AccelerateInfo::TriggeredByLoss(bool &is_triggered) {
  float target_loss = 0.0;
  REQUIRES_STATUS_OK(GetLossFromEnv(kTargetLoss, target_loss));
  float loss_to_change = GetLossToChange(target_loss, fast_ratio_);
  float loss_now = 0;
  REQUIRES_STATUS_OK(GetLossFromEnv(kLossNow, loss_now));
  ADP_LOG(INFO) << "[GEOP] accelerate train: get expected trigger recompile loss: " << loss_to_change
                << " with target loss: " << target_loss << " and loss now is: " << loss_now;
  if ((loss_now != 0.0) && (loss_now <= loss_to_change)) {
    ADP_LOG(EVENT) << "[GEOP] accelerate train: trigger recompile when loss is " << loss_now;
    if (loss_now != loss_to_change) {
      ADP_LOG(WARNING) << "[GEOP] accelerate train: trigger recompile loss smaller than expected loss, may"
                          " have some effect on train";
    }
    is_triggered = true;
    is_recovered_ = true;
    return Status::OK();
  }
  is_triggered = false;
  return Status::OK();
}

Status GeOp::AccelerateInfo::JudgeNeedRecompile(bool &need_recompile) {
  if (is_recovered_) {
    need_recompile = false;
    return Status::OK();
  }
  if (fast_mode_ == kModeValueStep) {
    REQUIRES_STATUS_OK(TriggeredByStep(need_recompile));
  } else {
    REQUIRES_STATUS_OK(TriggeredByLoss(need_recompile));
  }
  return Status::OK();
}

Status GeOp::DoAccelerateTrain() {
  if (!IsAccelerateTrainOn()) {
    return Status::OK();
  }
  // accelerate_train_mode_ must be valid if `IsAccelerateTrainOn` is true
  REQUIRES_STATUS_OK(ParserAccelerateTrain(accelerate_train_mode_));

  // accelerate by modify precision mode
  if (need_recover_precision_mode_) {
    REQUIRES_STATUS_OK(RecoverPrecisionMode());
  } else {
    REQUIRES_STATUS_OK(CheckAndModifyPrecisionMode());
  }
  return Status::OK();
}

Status GeOp::NeedRecompileWhenAccelerateTrainOn(bool &need_recompile) {
  if (!IsAccelerateTrainOn()) {
    need_recompile = false;
    return Status::OK();
  }
  REQUIRES_STATUS_OK(ParserAccelerateTrain(accelerate_train_mode_));
  return accelerate_info_.JudgeNeedRecompile(need_recompile);
}

Status GeOp::CheckAndSetAccelarateMode(const std::string &mode_value) {
  std::stringstream ss;
  if (valid_mode_values.find(mode_value) == valid_mode_values.end()) {
    const std::string valid_modes =
        std::accumulate(valid_mode_values.begin(), valid_mode_values.end(), std::string{},
                        [](const std::string &l, const std::string &r) { return l.empty() ? r : l + ", " + r; });
    ss << "accelerate_train_mode second part is invalid: " << mode_value << ", you can choose one of `" << valid_modes
       << "`";
    ADP_LOG(ERROR) << ss.str();
    return errors::Internal(ss.str());
  }
  if (mode_value == kModeValueStep) {
    uint32_t step = 0U;
    REQUIRES_STATUS_OK(GetStepFromEnv(kTotalStep, step));
    REQUIRES_STATUS_OK(GetStepFromEnv(kStepNow, step));
  }
  if (mode_value == kModeValueLoss) {
    float loss = 0.0;
    REQUIRES_STATUS_OK(GetLossFromEnv(kTargetLoss, loss));
    REQUIRES_STATUS_OK(GetLossFromEnv(kLossNow, loss));
  }
  accelerate_info_.fast_mode_ = mode_value;
  return Status::OK();
}

Status GeOp::CheckAndSetAccelarateRatio(const std::string &mode_value, const std::string &ratio_value) {
  float ratio = 0.0;
  std::stringstream ss;
  if (!strings::safe_strtof(ratio_value, &ratio)) {
    ss << "accelerate_train_mode third part is invalid: " << ratio_value
       << " ,you can choose `0.9` for `step` or `1.02` for `loss`";
    ADP_LOG(ERROR) << ss.str();
    return errors::Internal(ss.str());
  }

  if (mode_value == kModeValueStep) {
    if (ratio < kMinStepRatio || ratio > kMaxStepRatio) {
      ss << "accelerate_train_mode third part is invalid: " << ratio_value << " ,you can choose `" << kMinStepRatio
         << "-" << kMaxStepRatio << "` for `" << mode_value << "`";
      ADP_LOG(ERROR) << ss.str();
      return errors::Internal(ss.str());
    }
  } else if (mode_value == kModeValueLoss) {
    if (ratio < kMinLossRatio || ratio > kMaxLossRatio) {
      ss << "accelerate_train_mode third part is invalid: " << ratio_value << " ,you can choose `" << kMinLossRatio
         << "-" << kMaxLossRatio << "` for `" << mode_value << "`";
      ADP_LOG(ERROR) << ss.str();
      return errors::Internal(ss.str());
    }
  } else {
    ADP_LOG(ERROR) << "invalid mode value: " << mode_value;
    return errors::Internal("invalid mode value");
  }
  accelerate_info_.fast_ratio_ = ratio;
  return Status::OK();
}

Status GeOp::ParserAccelerateTrain(const std::string &accelerate_train_mode) {
  if (accelerate_info_.is_inited_) {
    return Status::OK();
  }
  accelerate_info_.iteration_per_loop_ = iteration_per_loop_;
  // format like "fast|step|0.9" or "fast|step"
  std::vector<std::string> infos = tensorflow::StringUtils::Split(accelerate_train_mode, '|');
  std::stringstream ss;
  if ((infos.size() != 2U) && (infos.size() != 3U)) {
    ss << "Format of accelerate_train_mode is invalid: " << accelerate_train_mode;
    ADP_LOG(ERROR) << ss.str();
    return errors::Internal(ss.str());
  }
  const auto &fast_value = infos[0U];
  const auto &iter = fast_value_string_2_eunm.find(fast_value);
  if (iter == fast_value_string_2_eunm.end()) {
    const std::string valid_values =
        std::accumulate(fast_value_string_2_eunm.begin(), fast_value_string_2_eunm.end(), std::string{},
                        [](const std::string &l, const std::pair<std::string, GeOp::FastValue> &r) {
                          return l.empty() ? r.first : l + ", " + r.first;
                        });
    ss << "accelerate_train_mode first part is invalid: " << fast_value << ", you can choose one of `" << valid_values
       << "`";
    ADP_LOG(ERROR) << ss.str();
    return errors::Internal(ss.str());
  }
  accelerate_info_.fast_value_ = iter->second;
  REQUIRES_STATUS_OK(CheckAndSetAccelarateMode(infos[1U]));
  if ((infos.size() != 3U) || (infos[2U].empty())) {
    accelerate_info_.fast_ratio_ =
        accelerate_info_.fast_mode_ == kModeValueStep ? kDefaultStepRatio : kDefaultLossRatio;
    accelerate_info_.is_inited_ = true;
    return Status::OK();
  }
  REQUIRES_STATUS_OK(CheckAndSetAccelarateRatio(accelerate_info_.fast_mode_, infos[2U]));
  accelerate_info_.is_inited_ = true;
  return Status::OK();
}

bool GeOp::IsAccelerateTrainOn() {
  return !(accelerate_train_mode_.empty());
}

Status GeOp::CheckAndModifyPrecisionMode() {
  std::stringstream ss;
  const auto &iter_v2 = init_options_.find(ge::PRECISION_MODE_V2);
  if ((accelerate_info_.origin_precision_mode_v2.empty()) && (iter_v2 != init_options_.end())) {
    const auto &origin_mode_v2 = init_options_[ge::PRECISION_MODE_V2];
    const auto &inner_iter_v2 = fast_value_2_precision_mode_v2.find(accelerate_info_.fast_value_);
    if ((inner_iter_v2 == fast_value_2_precision_mode_v2.end()) ||
        (supported_origin_precision_mode_v2.find(origin_mode_v2) == supported_origin_precision_mode_v2.end())) {
      ss << "accelerate fast_value:" << fast_value_enum_2_string.at(accelerate_info_.fast_value_)
         << " is not supported with PRECISION_MODE_V2: " << origin_mode_v2;
      ADP_LOG(ERROR) << ss.str();
      return errors::Internal(ss.str());
    }
    graph_options_[ge::PRECISION_MODE_V2] = inner_iter_v2->second;
    accelerate_info_.origin_precision_mode_v2 = origin_mode_v2;
    ADP_LOG(INFO) << "[GEOP] tf session " << tf_session_
                  << " change PRECISION_MODE_V2 from: " << accelerate_info_.origin_precision_mode_v2
                  << " to: " << inner_iter_v2->second;
    return Status::OK();
  }
  if ((accelerate_info_.origin_precision_mode_v1.empty())) {
    // if init_options_ has no PRECISION_MODE, set empty to origin mode
    const auto &origin_mode_v1 = init_options_[ge::PRECISION_MODE];
    const auto &inner_iter_v1 = fast_value_2_precision_mode_v1.find(accelerate_info_.fast_value_);
    if ((inner_iter_v1 == fast_value_2_precision_mode_v1.end()) ||
        (supported_origin_precision_mode_v1.find(origin_mode_v1) == supported_origin_precision_mode_v1.end())) {
      ss << "accelerate fast_value:" << fast_value_enum_2_string.at(accelerate_info_.fast_value_)
         << " is not supported with PRECISION_MODE: " << origin_mode_v1;
      ADP_LOG(ERROR) << ss.str();
      return errors::Internal(ss.str());
    }
    graph_options_[ge::PRECISION_MODE] = inner_iter_v1->second;
    accelerate_info_.origin_precision_mode_v1 = origin_mode_v1;
    ADP_LOG(INFO) << "[GEOP] tf session " << tf_session_
                  << " change PRECISION_MODE from: " << accelerate_info_.origin_precision_mode_v1
                  << " to: " << inner_iter_v1->second;
  }
  return Status::OK();
}

Status GeOp::RecoverPrecisionMode() {
  if (!accelerate_info_.origin_precision_mode_v2.empty()) {
    const auto fast_value = graph_options_[ge::PRECISION_MODE_V2];
    graph_options_[ge::PRECISION_MODE_V2] = accelerate_info_.origin_precision_mode_v2;
    ADP_LOG(INFO) << "[GEOP] tf session " << tf_session_ << " recover PRECISION_MODE_V2 from: " << fast_value
                  << " to: " << accelerate_info_.origin_precision_mode_v2;
  } else {
    const auto fast_value = graph_options_[ge::PRECISION_MODE];
    graph_options_[ge::PRECISION_MODE] = accelerate_info_.origin_precision_mode_v1;
    ADP_LOG(INFO) << "[GEOP] tf session " << tf_session_ << " recover PRECISION_MODE from: " << fast_value
                  << " to: " << accelerate_info_.origin_precision_mode_v1;
  }
  return Status::OK();
}

bool GeOp::IsGraphNeedRebuild(const uint32_t cache_graph_id) {
  if (NeedRecompileWhenAccelerateTrainOn(need_recover_precision_mode_) != Status::OK()) {
    ADP_LOG(ERROR) << "[GEOP] tf session " << tf_session_ << ", graph id: " << cache_graph_id
                   << " prepare to accelerate for train failed";
    return false;
  }
  return ((need_recover_precision_mode_) || (ge_session_->IsGraphNeedRebuild(cache_graph_id)));
}

bool GeOp::IncrementGraphIdCount(uint32_t &graph_id) {
  if (tf_session_.empty()) {
    ADP_LOG(ERROR) << "[GEOP] Add graph id failed, tf session is empty.";
    LOG(ERROR) << "[GEOP] Add graph id failed, tf session is empty.";
    return false;
  }
  auto it = session_and_graph_id_map_.find(tf_session_);
  if (it != session_and_graph_id_map_.end()) {
    it->second = it->second + kMaxCacheNum;
    graph_id = it->second;
    return true;
  }
  graph_id = 1;
  session_and_graph_id_map_.insert(std::make_pair(tf_session_, graph_id));
  return true;
}

bool GeOp::DecrementGraphIdCount(uint32_t &graph_id) {
  if (tf_session_.empty()) {
    ADP_LOG(ERROR) << "[GEOP] Sub graph id failed, tf session is empty.";
    LOG(ERROR) << "[GEOP] Sub graph id failed, tf session is empty.";
    return false;
  }

  auto it = session_and_graph_id_map_.find(tf_session_);
  if (it != session_and_graph_id_map_.end()) {
    if (it->second == 1) {
      it->second = it->second - 1;
      graph_id = it->second;
      return true;
    }
    it->second = it->second - kMaxCacheNum;
    graph_id = it->second;
    return true;
  }
  ADP_LOG(ERROR) << "[GEOP] Sub graph id failed, can not find tf session " << tf_session_;
  LOG(ERROR) << "[GEOP] Sub graph id failed, can not find tf session " << tf_session_;
  return false;
}

void GeOp::ClearGraphIdCount() {
  auto it = session_and_graph_id_map_.find(tf_session_);
  if (it != session_and_graph_id_map_.end()) {
    session_and_graph_id_map_.erase(it);
  }
}

void GeOp::GetExecGraphId(uint32_t &cache_graph_id, const std::vector<std::string> &input_shapes) {
  size_t num = cache_graphs_.size();
  if (cache_graphs_.find(input_shapes) != cache_graphs_.end()) {
    auto iter = std::find_if(graph_counts_.begin(), graph_counts_.end(),
                             [&input_shapes](const std::pair<std::vector<std::string>, uint32_t> graph_count) {
                               return graph_count.first == input_shapes;
                             });
    if (iter != graph_counts_.end()) {
      iter->second += 1;
    }
    cache_graph_id = cache_graphs_[input_shapes];
    ADP_LOG(INFO) << "Set graph_status to CompileDone when get exec graphid, graph_id: " << cache_graph_id;
    graph_handler_.status = CompileDone;
    graph_handler_.cv.notify_all();
  } else {
    ADP_LOG(INFO) << "[GEOP] This is a dynamic shape neural network, we recommend setting jit_compile to false";
    if (num >= kMaxCacheNum) {
      ADP_LOG(INFO) << "[GEOP] the cache vector size is : " << num << " , begin erase the least used";
      std::sort(graph_counts_.begin(), graph_counts_.end(), CmpValue);
      uint32_t erased_graph_id = cache_graphs_[graph_counts_[0].first];
      cache_graphs_.erase(graph_counts_[0].first);
      graph_counts_.erase(graph_counts_.cbegin());
      ge::Status status = ge_session_->RemoveGraph(erased_graph_id);
      if (status != ge::SUCCESS) {
        ADP_LOG(WARNING) << "[GEOP] GE Remove Graph failed, ret : " << ToString(status);
        LOG(WARNING) << "[GEOP] GE Remove Graph failed, ret : " << ToString(status);
      }
      cache_graph_id = erased_graph_id;
    } else {
      cache_graph_id = graph_id_ + num;
    }
    ADP_LOG(INFO) << "Set graph_status to Init when has no cache graph, graph_id: " << cache_graph_id;
    is_empty_graph_ = false;
    graph_handler_.status = Init;
    graph_handler_.cv.notify_all();
  }
}

bool GeOp::IsDynamicConfig() {
  const bool result = !graph_options_["ge.inputShape"].empty() && !graph_options_["ge.dynamicDims"].empty() &&
                      !graph_options_["ge.dynamicNodeType"].empty();
  ADP_LOG(INFO) << "[GEOP] IsDynamicConfig result is: " << result;
  return result;
}

void GeOp::SetDynamicInput() {
  if (dynamic_input_ == "1") {
    graph_options_["ge.exec.dynamicInput"] = dynamic_input_;
    graph_options_["ge.exec.dynamicGraphExecuteMode"] = dynamic_graph_execute_mode_;
    graph_options_["ge.exec.dataInputsShapeRange"] = data_inputs_shape_range_;
    if (dynamic_graph_execute_mode_ == "dynamic_execute" && data_inputs_shape_range_.empty() &&
        getnext_inputs_shape_range_.empty()) {
      graph_options_["ge.shape_generalized_build_mode"] = "shape_generalized";
    }
  }
}

PartialTensorShape GeOp::MakeCompatShape(const PartialTensorShape &a, const PartialTensorShape &b) const {
  const static auto kUnknownRankShape = PartialTensorShape();
  if (a.dims() != b.dims()) {
    return kUnknownRankShape;
  }
  return MakeUnknownShape(b.dims());
}

PartialTensorShape GeOp::MakeAdaptiveShape(const PartialTensorShape &a, const PartialTensorShape &b) const {
  const static auto kUnknownRankShape = PartialTensorShape();
  if (a.dims() != b.dims()) {
    return kUnknownRankShape;
  }
  static constexpr int64 kUnknownDim = -1;
  std::vector<int64> dims(a.dims(), kUnknownDim);
  for (int32_t i = 0; i < a.dims(); i++) {
    if (a.dim_size(i) == b.dim_size(i)) {
      dims[i] = a.dim_size(i);
    }
  }
  PartialTensorShape out_shape;
  auto status = PartialTensorShape::MakePartialShape(dims.data(), static_cast<int32_t>(dims.size()), &out_shape);
  return status.ok() ? out_shape : kUnknownRankShape;
}

void GeOp::InitGraphShape(OpKernelContext *const ctx) {
  mutex_lock lock{graph_handler_.graph_mu};
  for (size_t i = 0UL; i < static_cast<size_t>(ctx->num_inputs()); i++) {
    auto &shape = input_shapes_vec_[i];
    auto &value_shape = ctx->input(static_cast<int32_t>(i)).shape();
    if (!shape.has_value()) {
      // 第一次迭代时初始化shape
      if (compile_dynamic_mode_ == "1") {
        shape = MakeUnknownShape(value_shape.dims());
      } else {
        shape = value_shape;
      }
      ADP_LOG(INFO) << "Init input " << i << " shape to " << shape.value().DebugString();
    }
  }
}

bool GeOp::MaybeUpdateShape(OpKernelContext *const ctx) {
  ADP_LOG(INFO) << "MaybeUpdateShape, compile_dynamic_mode: " << compile_dynamic_mode_
                << ", jit_compile: " << jit_compile_ << ", shape_generalization_mode: " << shape_generalization_mode_;
  bool updated = false;
  for (size_t i = 0UL; i < static_cast<size_t>(ctx->num_inputs()); i++) {
    auto &shape = input_shapes_vec_[i];
    auto &value_shape = ctx->input(static_cast<int32_t>(i)).shape();
    if (!shape.value().IsCompatibleWith(value_shape)) {
      ADP_LOG(INFO) << "Compat input " << i << " shape " << shape.value().DebugString() << " vs. "
                    << value_shape.DebugString();
      updated = true;
      if (compile_dynamic_mode_ != "1" && jit_compile_ == "1" && shape_generalization_mode_ == "STRICT") {
        shape = value_shape;
        ADP_LOG(WARNING) << "Dynamic shape, recommended to configure jit_compile value to false or auto";
      } else if (compile_dynamic_mode_ != "1" && jit_compile_ == "1" && shape_generalization_mode_ == "ADAPTIVE") {
        shape = MakeAdaptiveShape(shape.value(), value_shape);
      } else {
        shape = MakeCompatShape(shape.value(), value_shape);
      }
      ADP_LOG(INFO) << "Refresh input " << i << " shape to " << shape.value().DebugString();
    }
  }
  return updated;
}

Status GeOp::CreateGeSession() {
  mutex_lock lock{mu_};
  if (sess_init_flag_) {
    return Status::OK();
  }
  // create ge session should be ensure after getinit async success
  const auto init_status = GePlugin::GetInstance()->GetInitStatus();
  const auto &warning_message = GePlugin::GetInstance()->GetInitWarningMessage();
  if (!warning_message.empty()) {
    LOG(WARNING) << "[GePlugin] GEInitialize warning message: " << std::endl << warning_message;
  }
  if (init_status != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GePlugin] Initialize ge failed, ret : " << ToString(init_status);
    const auto &error_message = GePlugin::GetInstance()->GetInitErrorMessage();
    std::stringstream ss;
    ss << "[GePlugin] Initialize ge failed, ret : " << ToString(init_status) << std::endl
       << "Error Message is : " << std::endl
       << error_message;
    LOG(ERROR) << ss.str();
    return errors::Internal(ss.str());
  }
  static bool first = true;
  if (first) {
    ADP_LOG(INFO) << "[GePlugin] Initialize ge success.";
    first = false;
  }
  if (!SessionManager::GetInstance().GetOrCreateGeSession(tf_session_, ge_session_, sess_options_) ||
      tf_session_.empty() || ge_session_ == nullptr) {
    return errors::Internal("Get ge session failed.");
  }
  sess_init_flag_ = true;
  ADP_LOG(INFO) << "[GEOP] tf session: " << tf_session_ << " get ge session success.";
  return Status::OK();
}

Status GeOp::DoGraphParser(ge::ComputeGraphPtr &compute_graph, FunctionLibraryDefinition *flib_def,
                           GraphDef &ori_graph_def) {
  std::map<ge::AscendString, ge::AscendString> const_value_map;
  std::vector<ge::AscendString> partition_graph;
  auto tf_status = SeparateGraphDef(ori_graph_def, partition_graph, const_value_map);
  if (!tf_status.ok()) {
    return tf_status;
  }
  auto build_sub_graph = [this, flib_def](const ge::AscendString &graph) -> ge::AscendString {
    const auto &graph_def = this->BuildSubGraph(flib_def, std::string(graph.GetString()));
    return ge::AscendString(graph_def.c_str(), graph_def.length());
  };
  ge::Status status =
      GeApiWrapper_ParseProtoWithSubgraph(partition_graph, const_value_map, build_sub_graph, compute_graph);
  if (status != ge::SUCCESS) {
    std::stringstream ss;
    ss << "graph parse failed. ret : " << status << std::endl
       << "Error Message is : " << std::endl
       << ge::GEGetErrorMsgV2().GetString();
    return errors::Internal(ss.str());
  }

  GeApiWrapper_SetDomiFormatFromParserContext();
  return Status::OK();
}

PartialTensorShape GeOp::MakeUnknownShape(const int32_t &size) const {
  const static auto kUnknownRankShape = PartialTensorShape();
  static constexpr int64 kUnknownDim = -1;
  std::vector<int64> dims(size, kUnknownDim);
  PartialTensorShape out_shape;
  auto status = PartialTensorShape::MakePartialShape(dims.data(), static_cast<int32_t>(dims.size()), &out_shape);
  return status.ok() ? out_shape : kUnknownRankShape;
}

Status GeOp::ParserGraph(OpKernelContext *ctx, const std::vector<Tensor> &input_vec) {
  // Get Graph
  auto func_lib = ctx->function_library();
  if (func_lib == nullptr) {
    return errors::Internal("function library is nullptr");
  }
  FunctionLibraryDefinition *flib_def =
      const_cast<FunctionLibraryDefinition *>(func_lib->GetFunctionLibraryDefinition());
  if (flib_def == nullptr) {
    return errors::Internal("flib_def is nullptr");
  }
  // Build GraphDef from FunctionDef
  GraphDef ori_graph_def;
  bool is_allreduce = false;
  auto ret = BuildGraphDef(*flib_def, input_vec, ori_graph_def, is_initialized_graph_, is_allreduce);
  if (!ret.ok()) {
    return ret;
  }
  if (kDumpGraph) {
    const std::string pbtxt_path = GetDumpPath() + "TF_" + ctx->op_kernel().name().c_str() + ".pbtxt";
    (void)WriteTextProto(Env::Default(), pbtxt_path, ori_graph_def);
  }
  ADP_LOG(INFO) << "[GEOP] TFadapter process graph success, GE parser begin, kernel_name: " << ctx->op_kernel().name()
                << " , tf session: " << tf_session_;
  const std::string compute_graph_name = "ge_default_" + CurrentTimeInStr();
  graph_handler_.graph = GeApiWrapper_MakeComputeGraphPtr(compute_graph_name.c_str());
  if (graph_handler_.graph == nullptr) {
    return errors::Internal("compute graph is nullptr");
  }
  // parser,  tensorflow graph to ge graph
  ret = DoGraphParser(graph_handler_.graph, flib_def, ori_graph_def);
  if (!ret.ok()) {
    return ret;
  }
  ADP_LOG(INFO) << "[GEOP] Tensorflow graph parse to ge graph success, kernel_name: " << ctx->op_kernel().name()
                << ", tf session: " << tf_session_ << ", iteration_per_loop: " << iteration_per_loop_
                << ", need iteration: " << need_iteration_;
  return SetGraphOptions(ctx);
}

Status GeOp::AddGraph(OpKernelContext *ctx, const uint32_t &graph_id) {
  // call ge session addGraph api
  auto graph_options = graph_options_;
  const auto it = graph_options.find("ge.inputShape");
  if (it != graph_options.end()) {
    // when some input is scale, input_shape option is changed to input_name:0 according to tfa guide,
    // here replace to input_name: to ensure it is scale shape [] instead of empty shape [0]
    RewriteInputShapeOption(it->second);
  }
  if (is_aoe_) {
    graph_options["ge.buildMode"] = "normal";
  }
  SetReuseOptions("ge.exec.inputReuseMemIndexes", ctx->num_inputs(), sess_options_, init_options_, graph_options);
  SetReuseOptions("ge.exec.outputReuseMemIndexes", ctx->num_outputs(), sess_options_, init_options_, graph_options);
  ADP_LOG(EVENT) << "[GEOP] call ge session add graph jit_compile: " << jit_compile_ << ", graph_id: " << graph_id;
  graph_options["ge.exec.graphIOMemAllocMode"] = "ByGE";

  const auto graph_option_ascend_string = ChangeStringToAscendString(graph_options);
  ADP_LOG(INFO) << "Graph options: ";
  NpuAttrs::LogOptions(graph_options);
  ge::Graph ge_graph = GeApiWrapper_CreateGraphFromComputeGraph(graph_handler_.graph);
  MarkDataNodesAsHostTensor(ge_graph);
  if (iteration_per_loop_ > 1) {
    ge_graph.SetNeedIteration(need_iteration_);
  }
  auto status = ge_session_->AddGraph(graph_id, ge_graph, graph_option_ascend_string);
  std::stringstream ss;
  if (status != ge::SUCCESS) {
    ss << "[GEOP] call ge session add graph failed, kernel: " << ctx->op_kernel().name()
       << ", tf session: " << tf_session_ << ", graph id: " << graph_id << std::endl
       << "Error Message is : " << std::endl
       << ge::GEGetErrorMsgV2().GetString();
    return errors::Internal(ss.str());
  }
  ADP_LOG(INFO) << "[GEOP] Add graph to ge session success, kernel_name: " << ctx->op_kernel().name()
                << ", tf session: " << tf_session_ << ", graph id: " << graph_id;
  return Status::OK();
}

Status GeOp::BuildGraph(const uint32_t &graph_id, const std::vector<ge::Tensor> &inputs) {
  // 由于GEInitialize跟geop是并行执行，需要等GEInitialize执行结束后才能打开profiling开关
  if (Profiler::GetInstance().IsEnabled()) {
    TF_RETURN_IF_ERROR(Profiler::GetInstance().Start());
  }
  ge::Status build_graph_status = ge_session_->BuildGraph(graph_id, inputs);
  std::stringstream ss;
  if (build_graph_status != ge::SUCCESS) {
    ss << "[GEOP] GE session build graph failed, domi_ret : " << build_graph_status << std::endl
       << "Error Message is : " << std::endl
       << ge::GEGetErrorMsgV2().GetString();
    return errors::Internal(ss.str());
  }
  LOG(INFO) << "The model has been compiled on the Ascend AI processor, current graph id is: " << graph_id;
  return Status::OK();
}

Status GeOp::RunGraph(OpKernelContext *ctx, const uint32_t &graph_id,
                      const std::shared_ptr<std::vector<ge::Tensor>> &inputs, DoneCallback done) {
  // 当非首次迭代打开profiling开关时，会跳过build阶段
  if (Profiler::GetInstance().IsEnabled()) {
    TF_RETURN_IF_ERROR(Profiler::GetInstance().Start());
  }
  // call ge session runGraphAsync api
  ADP_LOG(INFO) << "[GEOP] Call ge session RunGraphAsync, kernel_name: " << ctx->op_kernel().name()
                << ", tf session: " << tf_session_ << ", graph id: " << graph_id;
  int64_t run_start_time = InferShapeUtil::GetCurrentTimestap();
  // 因为inputs需要被ge使用，ge是异步接口，所以需要延长生命周期到回调函数结束
  auto callback = [done, ctx, run_start_time, inputs, this](ge::Status ge_status, std::vector<ge::Tensor> &outputs) {
    ExitCallbackGuarder guarder([this]() {
      mutex_lock lock(graph_handler_.graph_mu);
      ADP_LOG(INFO) << "Callback end, run_num: " << graph_handler_.graph_run_num;
      graph_handler_.graph_run_num--;
      graph_handler_.cv.notify_all();
    });
    if (ge_status == ge::SUCCESS) {
      if (BuildOutputTensorInfo(ctx, outputs) != Status::OK()) {
        ADP_LOG(ERROR) << ctx->op_kernel().name() << " GEOP::DoRunAsync get output failed.";
        std::stringstream ss;
        ss << ctx->op_kernel().name() << "GEOP::DoRunAsync get output failed." << std::endl
           << "Error Message is : " << std::endl
           << ge::GEGetErrorMsgV2().GetString();
        OP_REQUIRES_ASYNC(ctx, false, errors::Internal(ss.str()), done);
      }
    } else if (ge_status == ge::END_OF_SEQUENCE) {
      ctx->SetStatus(errors::OutOfRange("End of sequence"));
      ADP_LOG(WARNING) << "[GEOP] Out of range: End of sequence.";
      LOG(WARNING) << "[GEOP] Out of range: End of sequence.";
    } else if (ge_status != ge::SUCCESS) {
      ADP_LOG(ERROR) << ctx->op_kernel().name() << "GEOP::::DoRunAsync Failed";
      std::stringstream ss;
      ss << ctx->op_kernel().name() << "GEOP::::DoRunAsync Failed" << std::endl
         << "Error Message is : " << std::endl
         << ge::GEGetErrorMsgV2().GetString();
      OP_REQUIRES_ASYNC(ctx, false, errors::Internal(ss.str()), done);
    }
    int64_t run_end_time = InferShapeUtil::GetCurrentTimestap();
    ADP_LOG(INFO) << "[GEOP] RunGraphAsync callback, status:" << ge_status
                  << ", kernel_name:" << ctx->op_kernel().name() << "[ " << (run_end_time - run_start_time) << "us]";
    done();
  };
  const std::string geop_name = ctx->op_kernel().name();
  ge::Status run_graph_status = ge_session_->RunGraphAsync(graph_id, *inputs, callback);
  std::stringstream ss;
  if (run_graph_status != ge::SUCCESS) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kFatalSleepTime));
    ss << "[GEOP] call ge session RunGraphAsync Failed, kernel:" << geop_name << ", tf session: " << tf_session_
       << ", graph id: " << graph_id << std::endl
       << "Error Message is : " << std::endl
       << ge::GEGetErrorMsgV2().GetString();
    return errors::Internal(ss.str());
  }
  graph_handler_.graph_run_num++;
  ADP_LOG(INFO) << "End RunGraph: " << geop_name << ", run_num: " << graph_handler_.graph_run_num;
  return Status::OK();
}

Status GeOp::SetGraphOptions(OpKernelContext *ctx) {
  // convert to ge::graph
  if (graph_options_.count("input_format") != 0) {
    ADP_LOG(INFO) << "graph_options_[\"input_format\"] = " << graph_options_["input_format"];
  }

  if (iteration_per_loop_ > 1) {
    graph_options_["iterations_per_loop"] = std::to_string(iteration_per_loop_);
  }

  const auto cahce_option_iter = sess_options_.find("ge.graph_compiler_cache_dir");
  if (cahce_option_iter != sess_options_.cend() && !cahce_option_iter->second.empty()) {
    graph_options_["ge.graph_key"] = ctx->op_kernel().name();
  }

  if (is_host_graph_) {
    ADP_LOG(INFO) << "[GEOP] set graph option.";
    graph_options_["ge.exec.placement"] = "HOST";
  }
  graph_options_["ge.shape_generalized_build_mode"] = "shape_precise";
  if (!recompute_mode_.empty()) {
    graph_options_["ge.recompute"] = recompute_mode_;
  }
  SetDynamicInput();
  graph_options_["ge.exec.isVarInitGraph"] = is_var_init_graph_;
  graph_options_["ge.jit_compile"] = jit_compile_;
  graph_options_["ge.exec.overflow"] = "1";
  graph_options_["ge.graphLevelSat"] = (mix_compile_mode_ == "0") ? "1" : "0";
  return DoAccelerateTrain();
}

Status GeOp::CompileGraph(OpKernelContext *ctx, const std::vector<Tensor> &input_vec,
                          const std::vector<ge::Tensor> &inputs, const uint32_t &graph_id) {
  auto ret = ParserGraph(ctx, input_vec);
  if (!ret.ok()) {
    return ret;
  }
  // 初始化图直接返回
  if (is_initialized_graph_) {
    Tensor initialized_tensor(ctx->expected_output_dtype(0), TensorShape({0}));
    ctx->set_output(0, initialized_tensor);
    ADP_LOG(INFO) << "[GEOP] End GeOp::ComputeAsync, compute_graph is initialize, kernel_name:"
                  << ctx->op_kernel().name() << ", ret_status:" << ToString(ge::SUCCESS)
                  << " , tf session: " << tf_session_ << " ,graph id: " << graph_id;
    return Status::OK();
  }
  // 空图直接返回
  if (GeApiWrapper_GetAllNodesSize(graph_handler_.graph) == 0UL) {
    ADP_LOG(INFO) << "[GEOP] End GeOp::ComputeAsync, compute_graph is empty, kernel_name:" << ctx->op_kernel().name()
                  << ", ret_status:" << ToString(ge::SUCCESS) << " , tf session: " << tf_session_
                  << " ,graph id: " << graph_id;
    is_empty_graph_ = true;
    return Status::OK();
  }
  // 为了Parser与GeInitialize并行需求，将CreateGeSession放在此处
  ret = CreateGeSession();
  if (!ret.ok()) {
    return ret;
  }
  for (uint32_t i = 0U; i < graph_max_parallel_model_num_; i++) {
    ret = AddGraph(ctx, graph_id);
    if (!ret.ok()) {
      return ret;
    }
  }

  ret = BuildGraph(graph_id, inputs);
  if (!ret.ok()) {
    return ret;
  }
  return Status::OK();
}

Status GeOp::CompileAndRunGraph(OpKernelContext *ctx, const std::vector<Tensor> &input_vec,
                                const std::shared_ptr<std::vector<ge::Tensor>> &inputs,
                                const std::vector<std::string> &input_shapes, DoneCallback done) {
  mutex_lock lock{graph_handler_.graph_mu};

  bool is_life_control_enabled = ScopedGraphManager::Instance().IsControlEnabled();
  if (is_life_control_enabled) {
    ADP_LOG(INFO) << "[GEOP] Life control enabled, set graph options of const life cycle.";
    NPU_REQUIRES(ScopedGraphManager::Instance().SetGraph(tf_session_, graph_id_),
                 errors::Internal("Only support call sess.run once in scope of ScopedGraphManager."));
    graph_options_["ge.constLifecycle"] = "graph";
  }
  // 当其中一个线程处于compiling状态时，其他线程需要在此处wait，不能直接去编译
  while (graph_handler_.status == Compiling) {
    ADP_LOG(INFO) << "Compiling wait, graph_status: " << graph_handler_.status;
    graph_handler_.cv.wait(lock);
  }
  uint32_t cache_graph_id = graph_id_;
  if (IsLazyCompile()) {
    // in dynamic input mode, cache graphs.
    GetExecGraphId(cache_graph_id, input_shapes);
  }

  bool shape_changed = false;
  if ((!is_dynamic_input_) && (!IsDynamicConfig())) {
    shape_changed = MaybeUpdateShape(ctx);
  }

  // 判断是否shape发生变化或者需要重新编译
  if ((graph_handler_.status != Init) && (!is_empty_graph_) && (shape_changed || IsGraphNeedRebuild(cache_graph_id))) {
    ADP_LOG(INFO) << "[GEOP] The graph need rebuild, graph id " << cache_graph_id
                  << " , need_change_precision_mode: " << need_recover_precision_mode_;
    // 让进入需要Compiling状态时，其他线程需要等待此线程重编模型结束
    graph_handler_.status = Compiling;
    while (graph_handler_.graph_run_num > 0) {
      ADP_LOG(INFO) << "Remove wait, run_num: " << graph_handler_.graph_run_num
                    << ", graph_status: " << graph_handler_.status;
      graph_handler_.cv.wait(lock);
    }
    auto ret = ge_session_->RemoveGraph(cache_graph_id);
    if (ret != ge::SUCCESS) {
      // 防止remove报错时卡死
      ADP_LOG(INFO) << "Set graph_status to Init";
      graph_handler_.status = CompileDone;
      graph_handler_.cv.notify_all();
      return errors::Internal("[GEOP] Failed to remove graph ", cache_graph_id, " from ge, error code ", ret,
                              " Error Message is : ", ge::GEGetErrorMsgV2().GetString());
    }
    ADP_LOG(INFO) << "[GEOP] tf session: " << tf_session_ << ", graph id: " << cache_graph_id << " Removed graph";
  }

  if (graph_handler_.status != CompileDone) {
    auto ret = CompileGraph(ctx, input_vec, *inputs, cache_graph_id);
    ADP_LOG(INFO) << "Set graph_status to CompileDone";
    if (!ret.ok()) {
      graph_handler_.status = Init;
      graph_handler_.cv.notify_all();
      return ret;
    }
    graph_handler_.status = CompileDone;
    graph_handler_.cv.notify_all();
    if (need_compile_graph_first_) {
      // done函数执行后不能再使用ctx中的成员变量，否则会出现异常
      done();
      return Status::OK();
    }
  }

  if (is_initialized_graph_ || is_empty_graph_) {
    done();
    return Status::OK();
  }

  if (!IsDynamicConfig() && IsLazyCompile()) {
    cache_graphs_.insert(std::make_pair(input_shapes, cache_graph_id));
    graph_counts_.push_back(std::make_pair(input_shapes, 1));
  }

  return RunGraph(ctx, cache_graph_id, inputs, done);
}

bool GeOp::IsLazyCompile() {
  return ((dynamic_input_ == "1") && (dynamic_graph_execute_mode_ == "lazy_recompile"));
}

void GeOp::ComputeAsync(OpKernelContext *ctx, DoneCallback done) {
  ADP_LOG(INFO) << "[GEOP] Begin GeOp::ComputeAsync, kernel_name: " << ctx->op_kernel().name();
  int64_t start_time = InferShapeUtil::GetCurrentTimestap();
  const std::string geop_name = ctx->op_kernel().name();
  ExitCallbackGuarder guarder([start_time, geop_name]() {
    int64_t end_time = InferShapeUtil::GetCurrentTimestap();
    ADP_LOG(INFO) << "[GEOP] End GeOp::ComputeAsync, kernel_name: " << geop_name << ", cost ["
                  << ((end_time - start_time) / kMicrosToMillis) << "ms]";
  });
  OP_REQUIRES_ASYNC(ctx, init_flag_, errors::InvalidArgument("GeOp not Initialize success."), done);
  {
    mutex_lock lock{mu_};
    if (!graph_id_init_flag_) {
      if (job_type_ != "localhost") {  // in ps mode : ctx->session_handle() is empty
        tf_session_ = "ps_worker_session";
        ADP_LOG(INFO) << "[GEOP] get tf session " << tf_session_ << " when in ps mode.";
      }
      if (tf_session_.empty()) {
        tf_session_ = ctx->session_handle();
        ADP_LOG(INFO) << "[GEOP] Get tf session " << tf_session_ << " from session handle.";
      }
      OP_REQUIRES_ASYNC(ctx, IncrementGraphIdCount(graph_id_), errors::Internal("Get ge session failed."), done);
      graph_id_init_flag_ = true;
      ADP_LOG(INFO) << "[GEOP] Node name: " << ctx->op_kernel().name() << " , tf session: " << tf_session_;
    }
  }
  std::string env_profiling_mode;
  (void)ReadStringFromEnvVar("PROFILING_MODE", "", &env_profiling_mode);
  OP_REQUIRES_ASYNC(ctx,
                    !(((init_options_["profiling_mode"] == "1") || (env_profiling_mode == "true")) &&
                      (Profiler::GetInstance().IsEnabled())),
                    errors::InvalidArgument("The option 'profiling_mode' or env variables 'PROFILING_MODE' cannot be "
                                            "set to true when using 'profiler.Profiler'."),
                    done);
  if (is_aoe_) {
    ADP_LOG(INFO) << "[GEOP] In tuning func, aoe_mode:" << init_options_["ge.jobType"]
                  << ", work_path:" << init_options_["ge.tuningPath"]
                  << ", distribute_config:" << init_options_["distribute_config"];
    // aoe ini
    mutex_lock lock{mu_};
    if (!tuned_initialize_flag_) {
      std::map<ge::AscendString, ge::AscendString> global_options;
      global_options.insert({ge::AscendString("work_path"), ge::AscendString(init_options_["ge.tuningPath"].c_str())});
      global_options.insert({ge::AscendString("job_type"), ge::AscendString(init_options_["ge.jobType"].c_str())});
      global_options.insert({ge::AscendString("ge.resourceConfigPath"),
                             ge::AscendString(sess_options_["ge.resourceConfigPath"].c_str())});
      AoeStatus init_ret = (*aoe_initialize_)(global_options);
      OP_REQUIRES_ASYNC(ctx, init_ret == Aoe::AOE_SUCCESS,
                        errors::Internal("[GEOP] exec aoe initialize func failed[", init_ret, "]."), done);
      tuned_initialize_flag_ = true;
    }
  }
  // convert input to const
  OP_REQUIRES_OK_ASYNC(ctx, GraphInputConvertToConst(ctx), done);
  uint32_t num_inputs = static_cast<uint32_t>(ctx->num_inputs());
  ADP_LOG(INFO) << "[GEOP] Kernel_name:" << ctx->op_kernel().name() << ", num_inputs:" << num_inputs
                << ", num_outputs:" << ctx->num_outputs();

  // if input shapes changed, cache graphs
  std::vector<Tensor> input_vec;
  std::vector<std::string> input_shapes;
  std::shared_ptr<std::vector<ge::Tensor>> input_tensors = std::make_shared<std::vector<ge::Tensor>>();
  OP_REQUIRES_ASYNC(ctx, input_tensors != nullptr, errors::Internal("make shared input tensors failed"), done);
  OP_REQUIRES_OK_ASYNC(ctx, BuildInputTensorInfo(ctx, input_vec, input_shapes, *input_tensors), done);
  OP_REQUIRES_ASYNC(ctx, (!((is_dynamic_input_) && (compile_dynamic_mode_ == "1"))),
                    errors::Internal("Option compile_dynamic_mode cannot set when set dynamic_input to 1."), done);
  bool is_set_dynamic_config = IsDynamicConfig();
  OP_REQUIRES_ASYNC(
      ctx, (!((is_set_dynamic_config) && (compile_dynamic_mode_ == "1"))),
      errors::Internal(
          "Option compile_dynamic_mode cannot set when set input_shape, dynamic_dims and dynamic_node_type."),
      done);
  InitGraphShape(ctx);
  if (is_aoe_) {
    OP_REQUIRES_ASYNC(ctx, !is_set_dynamic_config, errors::Internal("Dynamic input config can not use with mstuning."),
                      done);
    auto input_vec_aoe = input_vec;
    OP_REQUIRES_ASYNC(ctx, RunTuning(input_vec_aoe, *input_tensors, ctx) == 0,
                      errors::Internal("RunTuning fail.\n", ge::GEGetErrorMsgV2().GetString()), done);
    ADP_LOG(INFO) << ctx->op_kernel().name() << " RunTuning finish.";
  }
  OP_REQUIRES_OK_ASYNC(ctx, CompileAndRunGraph(ctx, input_vec, input_tensors, input_shapes, done), done);
  return;
}

void GeOp::ChangeChannelNameAttr(NodeDef &node_def) const {
  const std::string pre_channel_name = node_def.attr().at("channel_name").s();
  uint32_t device_id = 0;
  (void)GetEnvDeviceID(device_id);
  AttrValue channel_name = AttrValue();
  channel_name.set_s(std::to_string(
      std::hash<std::string>{}(tf_session_ + pre_channel_name + "_device_" + std::to_string(device_id))));
  (*node_def.mutable_attr())["channel_name"] = channel_name;
  ADP_LOG(INFO) << "[GEOP] Changed the value of channel_name attr of node: " << node_def.name() << " to "
                << channel_name.s();
}

void GeOp::ProcessDpOpFuncDef(const Node &node) const {
  const std::string func_name = node.def().attr().at("function").func().name();
  const std::string org_func_def_lib = node.def().attr().at("func_def").s();
  FunctionDefLibrary func_def_lib;
  func_def_lib.ParseFromString(org_func_def_lib);
  bool is_new_transfer_mode = NpuAttrs::GetNewDataTransferFlag();
  for (auto &func_def : *func_def_lib.mutable_function()) {
    if (func_def.signature().name() == func_name) {
      for (auto &node_def : *func_def.mutable_node_def()) {
        if (!NpuAttrs::IsDatasetExecuteInDevice(tf_session_ + node_def.name()) &&
            (node_def.op() == "IteratorV2" || node_def.op() == "Iterator")) {
          NpuAttrs::SetDatasetExecuteInDeviceStatus(tf_session_ + node_def.name(), true);
        }
        if (node_def.op() == "DeviceQueueDataset") {
          if (is_new_transfer_mode) {
            ChangeChannelNameAttr(node_def);
          }
          tensorflow::AttrValue value;
          value.set_b(is_new_transfer_mode);
          node_def.mutable_attr()->insert({"_is_new_data_transfer", value});
        }
      }
    }
  }
  std::string new_func_def_lib;
  func_def_lib.SerializeToString(&new_func_def_lib);
  AttrValue func_def_value = AttrValue();
  func_def_value.set_s(new_func_def_lib);
  NodeDef &node_def = const_cast<NodeDef &>(node.def());
  (*node_def.mutable_attr())["func_def"] = func_def_value;
}

void GeOp::AddNodeAttrs(Node *node, bool &is_initialize) {
  // Add dp custom kernel label
  if (node->type_string() == "IteratorGetNext") {
    node->AddAttr("_kernel", "dp");
    if (dynamic_input_ == "1") {
      node->AddAttr("_dynamic_graph_execute_mode", dynamic_graph_execute_mode_);
      node->AddAttr("_getnext_inputs_shape_range", getnext_inputs_shape_range_);
    }
  }
  if (node->type_string() == "Assert" || node->type_string() == "Print" || node->type_string() == "PrintV2") {
    node->AddAttr("_kernel", "extend");
  }
  NodeDef &node_def = const_cast<NodeDef &>(node->def());
  if (node_def.op() == "Where") {
    is_initialize = InferShapeUtil::IsInitializedGraph(node);
  }
  if (node->name() == "IterationOp") {
    this->need_iteration_ = true;
    ADP_LOG(INFO) << "subgraph has iteration op.";
  }
  if (node->name().find("var_in_host") != std::string::npos) {
    is_host_graph_ = true;
    ADP_LOG(INFO) << "[GEOP] variable subgraph is initialized in host.";
  }
  if (!need_compile_graph_first_) {
    if (node->name().find("NpuCompile") != std::string::npos) {
      need_compile_graph_first_ = true;
      ADP_LOG(INFO) << "[GEOP] set subgraph compile first.";
    }
  }
  // clear device info && attr
  node_def.set_device("");
  if (node_def.op() == "Const") {
    node_def.mutable_attr()->erase("data_format");
    node_def.mutable_attr()->erase("cce_format");
    node_def.mutable_attr()->erase("output_type");
  }
}

void GeOp::BuildQueueDataAndGetNextFromQueue(Graph &graph, const Node &getnext_node,
                                             const std::string &channel_name) const {
  Node *get_next_from_queue = nullptr;
  Node *queue_data = nullptr;
  std::string get_next_from_queue_name = "get_next_from_queue_" + getnext_node.name();
  std::string queue_data_name = "queue_data_" + getnext_node.name();
  auto get_next_attrs = getnext_node.def().attr();
  TF_CHECK_OK(NodeBuilder(queue_data_name, "QueueData")
                  .Device(getnext_node.def().device())
                  .Attr("index", 0)
                  .Attr("T", DT_UINT8)
                  .Attr("queue_name", channel_name)
                  .Attr("output_types", get_next_attrs["output_types"])
                  .Attr("output_shapes", get_next_attrs["output_shapes"])
                  .Finalize(&graph, &queue_data));

  TF_CHECK_OK(NodeBuilder(get_next_from_queue_name, "GetNextFromQueue")
                  .Input(NodeBuilder::NodeOut(queue_data, 0))
                  .Device(getnext_node.def().device())
                  .Attr("output_types", get_next_attrs["output_types"])
                  .Attr("output_shapes", get_next_attrs["output_shapes"])
                  .Finalize(&graph, &get_next_from_queue));

  for (auto out_edge : getnext_node.out_edges()) {
    CHECK_NOT_NULL(out_edge);
    graph.AddEdge(get_next_from_queue, out_edge->src_output(), out_edge->dst(), out_edge->dst_input());
  }

  const OpDef &queue_data_op_def = queue_data->op_def();
  NodeDef &queue_data_node_def = const_cast<NodeDef &>(queue_data->def());
  std::string queue_data_op_def_string;
  queue_data_op_def.SerializeToString(&queue_data_op_def_string);
  tensorflow::AttrValue queue_data_attr;
  queue_data_attr.set_s(queue_data_op_def_string);
  queue_data_node_def.mutable_attr()->insert({"op_def", queue_data_attr});

  const OpDef &get_next_op_def = get_next_from_queue->op_def();
  NodeDef &get_next_node_def = const_cast<NodeDef &>(get_next_from_queue->def());
  std::string get_next_op_def_string;
  get_next_op_def.SerializeToString(&get_next_op_def_string);
  tensorflow::AttrValue get_next_attr;
  get_next_attr.set_s(get_next_op_def_string);
  get_next_node_def.mutable_attr()->insert({"op_def", get_next_attr});
}

bool GeOp::IsDynamicGetNext(const Node *node) {
  if (is_dynamic_input_) {
    return true;
  }
  auto it = is_getnext_dynamic_shape_.find(node->name());
  if (it == is_getnext_dynamic_shape_.end()) {
    return false;
  } else {
    return it->second;
  }
}

void GeOp::HandleDpOpAndGetNextNodes(Graph &graph) {
  std::vector<Node *> remove_nodes;
  for (Node *node : graph.nodes()) {
    CHECK_NOT_NULL(node);
    bool is_GetNext = (node->type_string() == "IteratorGetNext" || node->type_string() == "GetNext");
    if (node->type_string() == "DPOP") {
      ProcessDpOpFuncDef(*node);
    } else if (is_GetNext) {
      Node *iterator_node = nullptr;
      std::string iterator_name;
      NodeDef &node_def = const_cast<NodeDef &>(node->def());
      for (auto in_edge : node->in_edges()) {
        CHECK_NOT_NULL(in_edge);
        CHECK_NOT_NULL(in_edge->src());
        bool isIterator =
            (in_edge->src()->type_string() == "IteratorV2" || in_edge->src()->type_string() == "Iterator");
        if (isIterator) {
          iterator_name = in_edge->src()->name();
          iterator_node = in_edge->src();
        }
      }
      uint32_t device_id = 0;
      (void)GetDeviceID(device_id);
      std::string channel_name;
      if (HasNodeAttr(node->def(), "channel_name")) {
        channel_name = node->def().attr().at("channel_name").s();
      } else {
        channel_name = std::to_string(
            std::hash<std::string>{}(tf_session_ + iterator_name + "_device_" + std::to_string(device_id)));
      }
      ADP_LOG(DEBUG) << "[GEOP] channel_name:" << channel_name << ", device_id: " << device_id;

      if (kIsHeterogeneous) {
        BuildQueueDataAndGetNextFromQueue(graph, *node, channel_name);
        remove_nodes.push_back(node);
        if (iterator_node != nullptr) {
          remove_nodes.push_back(iterator_node);
        }
      } else if (NpuAttrs::IsDatasetExecuteInDevice(tf_session_ + iterator_name)) {
        if (IsDynamicGetNext(node)) {
          node_def.set_op("DynamicGetNext");
        }
      } else {
        Node *aicpu_getnext = nullptr;
        std::string aicpu_getnext_name = "aicpu_getnext_" + node->name();
        auto getnext_attrs = node->def().attr();
        std::string aicpu_getnext_type = IsDynamicGetNext(node) ? "DynamicGetNextV2" : "GetNext";
        TF_CHECK_OK(NodeBuilder(aicpu_getnext_name, aicpu_getnext_type)
                        .Device(node->def().device())
                        .Attr("channel_name", channel_name)
                        .Attr("output_types", getnext_attrs["output_types"])
                        .Attr("output_shapes", getnext_attrs["output_shapes"])
                        .Finalize(&graph, &aicpu_getnext));
        for (auto out_edge : node->out_edges()) {
          CHECK_NOT_NULL(out_edge);
          graph.AddEdge(aicpu_getnext, out_edge->src_output(), out_edge->dst(), out_edge->dst_input());
        }
        for (auto in_edge : node->in_edges()) {
          CHECK_NOT_NULL(in_edge);
          CHECK_NOT_NULL(in_edge->src());
          if (in_edge->IsControlEdge()) {
            graph.AddControlEdge(in_edge->src(), aicpu_getnext);
          }
        }
        const OpDef &getnext_op_def = aicpu_getnext->op_def();
        NodeDef &getnext_node_def = const_cast<NodeDef &>(aicpu_getnext->def());
        std::string op_def_s;
        getnext_op_def.SerializeToString(&op_def_s);
        tensorflow::AttrValue value;
        value.set_s(op_def_s);
        getnext_node_def.mutable_attr()->insert({"op_def", value});
        remove_nodes.push_back(node);
        if (iterator_node != nullptr) {
          remove_nodes.push_back(iterator_node);
        }
      }
      if (IsLazyCompile()) {
        graph_options_["ge.exec.enableCopyOutputAddr"] = "1";
      }
    }
  }
  for (Node *node : remove_nodes) {
    ADP_LOG(INFO) << "[GEOP] Remove node: " << node->name();
    graph.RemoveNode(node);
  }
}

Status GeOp::ProcessForDiffNodeTypes(Graph &graph, bool &is_initialize, bool &is_allreduce) {
  for (Node *node : graph.nodes()) {
    if (node->type_string() == kAllReduce) {
      is_allreduce = true;
    }
    AddNodeAttrs(node, is_initialize);
    // Add Input&Output Desc into NodeDef
    Status ret = this->GenerateDesc(node);
    if (!ret.ok()) {
      ADP_LOG(ERROR) << "[GEOP] node: " << node->name() << " GenerateDesc failed, " << ret.error_message();
      LOG(ERROR) << "[GEOP] node: " << node->name() << " GenerateDesc failed, " << ret.error_message();
      return ret;
    }

    if (node->type_string() == "NpuOnnxGraphOp") {
      ret = this->ParseOnnxGraphOpAttr(node);
      graph_options_["input_format"] = "NCHW";
      ADP_LOG(INFO) << "onnx_graph_parser graph_options_[\"input_format\"] = " << graph_options_["input_format"];
      if (!ret.ok()) {
        LOG(ERROR) << "[GEOP]node: " << node->name() << " Parse Node with Onnx Model failed, " << ret.error_message();
        return ret;
      }
    }

    if (node->type_string() == "IteratorGetNext" || node->type_string() == "GetNext") {
      ProcessGetNextNode(node);
    }
  }
  return Status::OK();
}

void GeOp::ProcessGetNextNode(const Node *node) {
  bool is_dynamic_shape = false;
  const char *kTypeAttrName = "output_types";
  const char *kShapeAttrName = "output_shapes";
  std::vector<DataType> type_attrs;
  std::vector<const TensorShapeProto *> shape_attrs;
  if (tensorflow::TryGetNodeAttr(node->attrs(), kShapeAttrName, &shape_attrs)) {
    for (auto i = 0; i < node->num_outputs(); i++) {
      const TensorShapeProto &shape_proto = *shape_attrs[i];
      tensorflow::PartialTensorShape shape(shape_proto);
      if (!shape.IsFullyDefined()) {
        is_dynamic_shape = true;
        ADP_LOG(INFO) << "[GEOP]node: " + node->name() + " is_dynamic_shape come true.";
      }
    }
  }
  if ((!is_dynamic_shape) && tensorflow::TryGetNodeAttr(node->attrs(), kTypeAttrName, &type_attrs)) {
    for (auto i = 0; i < node->num_outputs(); i++) {
      if (type_attrs[i] == DT_STRING) {
        is_dynamic_shape = true;
        ADP_LOG(INFO) << "[GEOP]node: " + node->name() + "'s output_types include DT_STRING.";
      }
    }
  }
  auto it = is_getnext_dynamic_shape_.find(node->name());
  if (it == is_getnext_dynamic_shape_.end()) {
    (void)is_getnext_dynamic_shape_.insert(std::make_pair(node->name(), is_dynamic_shape));
  } else {
    ADP_LOG(WARNING) << "[GEOP]node: " + node->name() + " has is_dynamic_shape[" << it->second << "].";
  }
}

void GeOp::UpdateInputsShapeDesc(Graph &graph) {
  for (auto node : graph.op_nodes()) {
    if (!node->IsArg()) {
      continue;
    }
    size_t index = static_cast<size_t>(node->attrs().Find("index")->i());
    node->ClearAttr("_output_shapes");
    if (!input_shapes_vec_[index].has_value()) {
      continue;
    }
    node->AddAttr("_output_shapes", std::vector<PartialTensorShape>{input_shapes_vec_[index].value()});
    NodeDef &node_def = const_cast<NodeDef &>(node->def());
    AttrValue &output_tensor_descs = (*node_def.mutable_attr())[OUTPUT_DESC];
    auto &shape = input_shapes_vec_[index].value();
    AttrValue attr_shape_value;
    attr_shape_value.set_type(DT_INT32);
    if (shape.unknown_rank()) {
      const int kUnknownRankDimSize = -2;
      attr_shape_value.mutable_list()->add_i(kUnknownRankDimSize);
    } else {
      for (auto i = 0; i < shape.dims(); ++i) {
        attr_shape_value.mutable_list()->add_i(shape.dim_size(i));
      }
    }
    (*output_tensor_descs.mutable_list()->mutable_func(0)->mutable_attr())[SERIALIZE_SHAPE] = attr_shape_value;
  }
}

// Build GraphDef from FunctionDef.
Status GeOp::BuildGraphDef(FunctionLibraryDefinition &flib_def, const std::vector<Tensor> &input_vec,
                           GraphDef &graph_def, bool &is_initialize, bool &is_allreduce) {
  const FunctionDef *function_def = flib_def.Find(function_.name());
  NPU_REQUIRES(function_def != nullptr, errors::Internal("Function:", function_.name(), " fdef is nullptr"));
  // get infershape
  Graph graph(OpRegistry::Global());
  Status ret = InferShapeUtil::InferShape(input_vec, &flib_def, function_def, &graph);
  if (!ret.ok()) {
    ADP_LOG(ERROR) << "[GEOP] InferShape failed, " << ret.error_message();
    LOG(ERROR) << "[GEOP] InferShape failed, " << ret.error_message();
    return ret;
  }

  std::vector<Node *> dynamic_shape_nodes;
  bool is_set_dynamic_config = IsDynamicConfig();
  if (is_set_dynamic_config) {
    jit_compile_ = "1";
    BuildShapeNodeAndCacheArgNodes(graph, dynamic_shape_nodes);
  }

  NPU_REQUIRES_OK(ProcessForDiffNodeTypes(graph, is_initialize, is_allreduce));

  // set input_shape to dynamic nodes shape desc
  if (is_set_dynamic_config) {
    ret = ChangeInputsShapeDesc(dynamic_shape_nodes);
    if (!ret.ok()) {
      ADP_LOG(ERROR) << "[GEOP] ChangeInputsShapeDesc failed, " << ret.error_message();
      LOG(ERROR) << "[GEOP] ChangeInputsShapeDesc failed, " << ret.error_message();
      return ret;
    }
  }
  HandleDpOpAndGetNextNodes(graph);

  // 动态场景下需要将动态轴更新成-1，避免频繁触发编译
  if ((jit_compile_ != "1") || (compile_dynamic_mode_ == "1") ||
      (jit_compile_ == "1" && shape_generalization_mode_ != "STRICT")) {
    ADP_LOG(INFO) << "[GEOP] UpdateInputsShapeDesc start.";
    UpdateInputsShapeDesc(graph);
  }

  graph.ToGraphDef(&graph_def);
  std::string enable_force_v2_control;
  (void)ReadStringFromEnvVar("ENABLE_FORCE_V2_CONTROL", "", &enable_force_v2_control);
  if (enable_force_v2_control == "1") {
    Status status = FunctionalizeControlFlow(&graph, &flib_def);
    if (status != Status::OK()) {
      LOG(WARNING) << "[GEOP] Failed functionalize control flow: " << status.error_message();
      return Status::OK();
    }
    graph.ToGraphDef(&graph_def);
  }
  return Status::OK();
}

Status GeOp::SeparateGraphDef(GraphDef &ori_graph_def, std::vector<ge::AscendString> &partition_graph,
                              std::map<ge::AscendString, ge::AscendString> &const_value_map) {
  std::string graph_def_str = ori_graph_def.SerializeAsString();
  if (!graph_def_str.empty()) {
    partition_graph.push_back(ge::AscendString(graph_def_str.c_str(), graph_def_str.length()));
    return Status::OK();
  }
  LOG(INFO) << "GraphDef exceeds 2G, weights need to be separated from the model";
  ADP_LOG(INFO) << "GraphDef exceeds 2G, weights need to be separated from the model";
  for (NodeDef &node : *ori_graph_def.mutable_node()) {
    if (node.op() == "Const") {
      std::string node_name = node.name();
      auto iter = node.mutable_attr()->find("value");
      if (iter == node.mutable_attr()->end()) {
        ADP_LOG(ERROR) << "Const node: " << node_name << " don't have value attribute";
        return errors::InvalidArgument("Const node don't have value attribute");
      }
      TensorProto *tensor = iter->second.mutable_tensor();
      std::string tensor_str = tensor->SerializeAsString();
      const_value_map.insert({ge::AscendString(node_name.c_str(), node_name.length()),
                              ge::AscendString(tensor_str.c_str(), tensor_str.length())});
      node.mutable_attr()->erase(iter);
    }
  }
  graph_def_str = ori_graph_def.SerializeAsString();
  partition_graph.push_back(ge::AscendString(graph_def_str.c_str(), graph_def_str.length()));
  return Status::OK();
}

Status GeOp::ParseOnnxGraphOpAttr(Node *&node) const {
  NodeDef &node_def = const_cast<NodeDef &>(node->def());

  // Get input and output numbers of NpuOnnxGraphOp op
  AttrValue in_value;
  int32_t inout_nums = node->num_inputs();
  in_value.set_i(static_cast<int32_t>(inout_nums));
  node_def.mutable_attr()->insert({"_input_num", in_value});
  inout_nums = node->num_outputs();
  AttrValue ot_value;
  ot_value.set_i(static_cast<int32_t>(inout_nums));
  node_def.mutable_attr()->insert({"_output_num", ot_value});

  std::string model_path = node_def.attr().find("model_path")->second.s();
  std::string graph_name = "onnx_compute_graph_" + node->name();
  ge::Graph sub_graph(graph_name.c_str());
  std::map<ge::AscendString, ge::AscendString> parser_params;
  std::string subgrph_name("onnx_compute_graph_" + node->name() + '_' + CurrentTimeInStr());
  parser_params.insert({ge::AscendString(ge::ir_option::OUTPUT), ge::AscendString(subgrph_name.c_str())});
  ge::Status status = ge::aclgrphParseONNX(model_path.c_str(), parser_params, sub_graph);
  if (status != ge::SUCCESS) {
    ADP_LOG(ERROR) << "[GEOP] node: " << node->name() << ": Onnx model parse failed, ret: " << ToString(status);
    std::stringstream ss;
    ss << "[GEOP] node: " << node->name() << ": Onnx model parse failed, ret: " << ToString(status) << std::endl
       << "Error Message is : " << std::endl
       << ge::GEGetErrorMsgV2().GetString();
    return errors::Internal(ss.str());
  }

  // rename the nodes in subgraph of onnx model
  GeApiWrapper_RenameAllNodes(&sub_graph, node->name().c_str());
  std::string model_str;
  GeApiWrapper_ModelSaveToString(sub_graph, node->name(), model_str);
  AttrValue attr_value;
  attr_value.set_s(model_str);
  node_def.mutable_attr()->insert({"_external_model", attr_value});
  return Status::OK();
}

void GeOp::BuildShapeNodeAndCacheArgNodes(Graph &graph, std::vector<Node *> &dynamic_shape_nodes) {
  if (kIsHeterogeneous) {
    ADP_LOG(INFO) << "Is heterogeneous, no need to build shape node and cache arg nodes.";
    return;
  }
  std::string dynamic_node_type = graph_options_["ge.dynamicNodeType"];
  for (Node *node : graph.nodes()) {
    // add shape node to get getnext node real shape
    if (dynamic_node_type == "0" && node->type_string() == "IteratorGetNext") {
      dynamic_shape_nodes.emplace_back(node);
      ADP_LOG(INFO) << "push in dynamic shape nodes, node: " << node->name() << ", type: " << node->type_string();
      std::set<int32_t> out_index;
      for (auto out_edge : node->out_edges()) {
        if (!out_edge->IsControlEdge()) {
          std::string msg = "Src:" + out_edge->src()->name() + ":" + std::to_string(out_edge->src_output()) +
                            ", Dst:" + out_edge->dst()->name() + ":" + std::to_string(out_edge->dst_input());
          ADP_LOG(INFO) << "[GEOP] GetNext node in out info : " << msg;
          out_index.insert(out_edge->src_output());
        }
      }
      for (int32_t idx : out_index) {
        std::string shape_name = "getnext_shape_" + std::to_string(idx);
        Node *shape_node = nullptr;
        TF_CHECK_OK(NodeBuilder(shape_name, "Shape")
                        .Input(node, idx)
                        .Device(node->def().device())
                        .Finalize(&graph, &shape_node));
        std::string identity_name = "shape_identity_" + std::to_string(idx);
        Node *identity_node = nullptr;
        TF_CHECK_OK(NodeBuilder(identity_name, "Identity")
                        .Input(shape_node, 0)
                        .Device(shape_node->def().device())
                        .Finalize(&graph, &identity_node));
      }
    }
    // count data args and getnext args for dynamic dims
    if (node->type_string() == "_Arg") {
      if (node->name().find("IteratorGetNext_") != std::string::npos) {
        if (dynamic_node_type == "0") {
          dynamic_shape_nodes.emplace_back(node);
          ADP_LOG(INFO) << "push in dynamic shape nodes, node : " << node->name() << ", type : " << node->type_string();
        }
      } else {
        if (dynamic_node_type == "1") {
          dynamic_shape_nodes.emplace_back(node);
          ADP_LOG(INFO) << "push in dynamic shape nodes, node: " << node->name() << ", type: " << node->type_string();
        }
      }
    }
  }
  // sort dynamic nodes to match input_shapes
  std::sort(dynamic_shape_nodes.begin(), dynamic_shape_nodes.end(), CmpVecValue);
}

Status GeOp::ChangeInputsShapeDesc(std::vector<Node *> &dynamic_shape_nodes) {
  if (kIsHeterogeneous) {
    ADP_LOG(INFO) << "Is heterogeneous, no need to change inputs shape desc.";
    return Status::OK();
  }
  std::vector<std::string> result;
  std::string input_shapes = graph_options_["ge.inputShape"];
  Split(input_shapes, result, ";");  // e.g. result:["data:2,3", "data1:3,4"]

  if (dynamic_shape_nodes.size() == 1U && dynamic_shape_nodes[0]->type_string() == "IteratorGetNext") {
    ADP_LOG(INFO) << "[GEOP] Change " << dynamic_shape_nodes[0]->name() << " shape desc.";
    if (dynamic_shape_nodes[0]->num_outputs() != static_cast<int32>(result.size())) {
      return errors::InvalidArgument("input_shape is not match inputs num in graph");
    }
    NodeDef &node_def = const_cast<NodeDef &>(dynamic_shape_nodes[0]->def());
    AttrValue &output_tensor_descs = (*node_def.mutable_attr())[OUTPUT_DESC];
    for (int32 i = 0; i < dynamic_shape_nodes[0]->num_outputs(); ++i) {
      AttrValue attr_shape_value;
      attr_shape_value.set_type(DT_INT32);
      SetShapesToOutputDesc(result, i, attr_shape_value);
      (*output_tensor_descs.mutable_list()->mutable_func(i)->mutable_attr())[SERIALIZE_SHAPE] = attr_shape_value;
    }
  } else {
    if (!dynamic_shape_nodes.empty()) {
      if (dynamic_shape_nodes.size() != result.size()) {
        return errors::InvalidArgument("input_shape is not match inputs num in graph");
      }
    }
    for (size_t i = 0U; i < dynamic_shape_nodes.size(); ++i) {
      ADP_LOG(INFO) << "[GEOP] Change " << dynamic_shape_nodes[i]->name() << " shape desc.";
      NodeDef &node_def = const_cast<NodeDef &>(dynamic_shape_nodes[i]->def());
      AttrValue &output_tensor_descs = (*node_def.mutable_attr())[OUTPUT_DESC];
      AttrValue attr_shape_value;
      attr_shape_value.set_type(DT_INT32);
      SetShapesToOutputDesc(result, i, attr_shape_value);
      (*output_tensor_descs.mutable_list()->mutable_func(0)->mutable_attr())[SERIALIZE_SHAPE] = attr_shape_value;
    }
  }
  ADP_LOG(INFO) << "[GEOP] change input shapes desc success.";
  return Status::OK();
}

void GeOp::SetShapesToOutputDesc(const std::vector<std::string> &input_shapes, const int &index,
                                 AttrValue &attr_shape_value) const {
  if (input_shapes.empty()) {
    ADP_LOG(ERROR) << "[GEOP] input_shapes is empty.";
    LOG(ERROR) << "[GEOP] input_shapes is empty.";
    return;
  }
  if (index < 0) {
    ADP_LOG(ERROR) << "[GEOP] index must be more than 0.";
    LOG(ERROR) << "[GEOP] index must be more than 0.";
    return;
  }
  ADP_LOG(INFO) << "[GEOP] Get input: " << index << ", input shape: " << input_shapes[index];
  std::vector<std::string> shape;
  Split(input_shapes[index], shape, ":");  // e.g. shape:["data", "2,3,4"]
  if (shape.empty() || shape.size() != 2) {
    ADP_LOG(ERROR) << "[GEOP] shape is empty or shape size is not 2.";
    LOG(ERROR) << "[GEOP] shape is empty or shape size is not 2.";
    return;
  }
  if (shape[1] == "0") {
    // scale node has no shape.
    return;
  }
  std::vector<std::string> dims;
  Split(shape[1], dims, ",");  // e.g. dims:["2", "3", "4"]
  for (auto dim : dims) {
    attr_shape_value.mutable_list()->add_i(std::atoi(dim.c_str()));
  }
}

int GeOp::RunTuning(std::vector<Tensor> &input_vec, std::vector<ge::Tensor> &inputs, const OpKernelContext *const ctx) {
  mutex_lock lock{graph_handler_.graph_mu};
  if (tuned_flag_.test_and_set()) {
    ADP_LOG(INFO) << ctx->op_kernel().name() << " has tuned.";
    return 0;
  }
  ADP_LOG(INFO) << "[GEOP] " << ctx->op_kernel().name() << " begin tune.";

  // Get Graph
  if (ctx->function_library() == nullptr) {
    ADP_LOG(ERROR) << "function library is nullptr";
    return -1;
  }
  FunctionLibraryDefinition *flib_def =
      const_cast<FunctionLibraryDefinition *>(ctx->function_library()->GetFunctionLibraryDefinition());
  if (flib_def == nullptr) {
    ADP_LOG(ERROR) << "flib_def is nullptr";
    return -1;
  }
  std::shared_ptr<Graph> graph = std::make_shared<Graph>(OpRegistry::Global());
  if (graph == nullptr) {
    ADP_LOG(ERROR) << "create tensorflow graph failed";
    return -1;
  }

  // Build GraphDef from FunctionDef
  bool is_allreduce = false;
  GraphDef ori_graph_def;
  Status s = BuildGraphDef(*flib_def, input_vec, ori_graph_def, is_initialized_graph_, is_allreduce);
  if (!s.ok()) {
    ADP_LOG(ERROR) << "BuildGraphDef error";
    return -1;
  }

  if (is_initialized_graph_) {
    ADP_LOG(INFO) << ctx->op_kernel().name() << " graph is initialized";
    return 0;
  }

  if (kDumpGraph) {
    const std::string pbtxt_path = GetDumpPath() + "TF_" + ctx->op_kernel().name() + "_AOE.pbtxt";
    (void)WriteTextProto(Env::Default(), pbtxt_path, ori_graph_def);
  }
  const std::string compute_graph_name = "ge_default_" + CurrentTimeInStr();
  ge::ComputeGraphPtr compute_graph = GeApiWrapper_MakeComputeGraphPtr(compute_graph_name.c_str());
  if (compute_graph == nullptr) {
    ADP_LOG(ERROR) << "create ComputeGraph failed";
    return -1;
  }
  // parser,  tensorflow graph to ge graph
  auto status = DoGraphParser(compute_graph, flib_def, ori_graph_def);
  if (!(status.ok())) {
    ADP_LOG(ERROR) << status.error_message();
    return -1;
  }
  ADP_LOG(INFO) << "[GEOP] Tensorflow graph parse to ge graph success.";

  // convert to ge::graph
  ge::Graph ge_graph = GeApiWrapper_CreateGraphFromComputeGraph(compute_graph);
  ge_graph.SetNeedIteration(false);
  if (is_host_graph_) {
    graph_options_["ge.exec.placement"] = "HOST";
  }
  SetDynamicInput();
  graph_options_["ge.exec.overflow"] = "1";
  graph_options_["ge.graphLevelSat"] = (mix_compile_mode_ == "0") ? "1" : "0";
  // run aoe tuning
  return ExecuteAoeTuning(ge_graph, is_allreduce, inputs);
}

int GeOp::ExecuteAoeTuning(ge::Graph &ge_graph, bool is_allreduce, std::vector<ge::Tensor> &inputs) {
  if ((init_options_["ge.jobType"] == "1") || (init_options_["ge.jobType"] == "2") ||
      ((init_options_["ge.jobType"] == "4") && is_allreduce)) {
    std::function<void()> callback = [this]() {
      if (aoe_destroy_session_ != nullptr) {
        AoeStatus aoe_destroy_ret = (*aoe_destroy_session_)(session_id_);
        if (aoe_destroy_ret != Aoe::AOE_SUCCESS) {
          ADP_LOG(ERROR) << "exec aoe destroy func failed[" << aoe_destroy_ret << "].";
          return;
        }
        ADP_LOG(INFO) << "[GEOP] aoe destroy success[" << aoe_destroy_ret << "].";
      }
    };
    ADP_LOG(INFO) << "[GEOP] in tune mode, training graph handled by tools.";

    // aoe create session
    AoeStatus session_ret = (*aoe_create_session_)(session_id_);
    if (session_ret != Aoe::AOE_SUCCESS) {
      ADP_LOG(ERROR) << "exec aoe create session func failed[" << session_ret << "].";
      return -1;
    }
    {
      GE_MAKE_GUARD(destroy, callback);
      const auto status = CreateGeSession();
      if (!status.ok()) {
        return -1;
      }
      // share ge_session to aoe
      AoeStatus set_ret = (*aoe_set_gesession_)(session_id_, ge_session_);
      if (set_ret != Aoe::AOE_SUCCESS) {
        ADP_LOG(ERROR) << "exec aoe set session func failed[" << set_ret << "].";
        return -1;
      }
      // set tuning graph
      AoeStatus tune_ret = (*aoe_set_tuninggraph_)(session_id_, ge_graph);
      if (tune_ret != Aoe::AOE_SUCCESS) {
        ADP_LOG(ERROR) << "exec aoe set graph func failed[" << tune_ret << "].";
        return -1;
      }
      // set tuning inputs
      AoeStatus set_inputs_ret = (*aoe_set_tuning_graph_input_)(session_id_, inputs);
      if (set_inputs_ret != Aoe::AOE_SUCCESS) {
        ADP_LOG(ERROR) << "exec aoe set tuning inputs func failed[" << set_inputs_ret << "].";
        return -1;
      }
      // aoe tuning
      std::map<ge::AscendString, ge::AscendString> tuing_options;
      tuing_options.insert({ge::AscendString("ge.recompute"), ge::AscendString(recompute_mode_.c_str())});
      tuing_options.insert(
          {ge::AscendString("ge.aoe_config_file"), ge::AscendString(init_options_["ge.aoe_config_file"].c_str())});
      AoeStatus aoe_tune_ret = (*aoe_tuning_graph_)(session_id_, tuing_options);
      if ((aoe_tune_ret != Aoe::AOE_SUCCESS) && (aoe_tune_ret != Aoe::AOE_ERROR_NON_OPTIMIZE_GRAPH)) {
        ADP_LOG(ERROR) << "exec aoe tuning func failed[" << aoe_tune_ret << "].";
        return -1;
      }
      ADP_LOG(INFO) << "[GEOP] Aoe success[" << aoe_tune_ret << "].";
    }
  }
  return 0;
}

std::string GeOp::BuildSubGraph(FunctionLibraryDefinition *flib_def, const std::string &graph) {
  ADP_LOG(INFO) << "[GEOP] build_sub_graph enter, sub graph name is " << graph;
  const FunctionDef *func_def = flib_def->Find(graph);
  if (func_def == nullptr) {
    ADP_LOG(ERROR) << "[GEOP] Sub graph not found in library, sub_graph_name: " << graph;
    return "";
  }
  // get infershape
  Graph subgraph(flib_def);
  Status status = InferShapeUtil::GetSubGraphFromFunctionDef(*flib_def, *func_def, &subgraph);
  if (status != Status::OK()) {
    ADP_LOG(ERROR) << "[GEOP] Get subgraph from functiondef fail:" << status.error_message();
    return "";
  }
  ADP_LOG(INFO) << "[GEOP] Get subgraph from functiondef success.";
  std::string enable_force_v2_control;
  (void)ReadStringFromEnvVar("ENABLE_FORCE_V2_CONTROL", "", &enable_force_v2_control);
  if (enable_force_v2_control == "1" && kDumpGraph) {
    GraphDef graph_def;
    subgraph.ToGraphDef(&graph_def);
    WriteTextProto(Env::Default(), GetDumpPath() + graph + "_graph.pbtxt", graph_def);
  }
  bool is_initialize = false;
  for (Node *node : subgraph.nodes()) {
    AddNodeAttrs(node, is_initialize);
    // Add Input&Output Desc into NodeDef
    if (GenerateDesc(node) != Status::OK()) {
      ADP_LOG(WARNING) << "[GEOP] name: " << node->name() << " op:" << node->type_string()
                       << " Generate desc failed in subgraph.";
    }
  }
  std::unique_ptr<GraphDef> sub_graph_def(new (std::nothrow) GraphDef());
  if (sub_graph_def == nullptr) {
    ADP_LOG(ERROR) << "[GEOP] Malloc memory for subgraph def fail.";
    return "";
  }
  subgraph.ToGraphDef(sub_graph_def.get());
  if (enable_force_v2_control == "1") {
    sub_graph_def->clear_library();
    sub_graph_def->mutable_versions()->clear_min_consumer();
  }

  if (kDumpGraph) {
    const std::string pbtxt_path = GetDumpPath() + "TF_Subgraph_" + graph.c_str() + ".pbtxt";
    (void)WriteTextProto(Env::Default(), pbtxt_path, *sub_graph_def);
  }
  ADP_LOG(INFO) << "[GEOP] build_sub_graph exit, sub_graph_name : " << graph;
  return sub_graph_def->SerializeAsString();
}

void GeOp::AnalyzeInputDesc(bool need_collect_shapes, void *tensor_ptr, ge::Tensor &input, ge::DataType type,
                            std::vector<std::string> &input_shapes) const {
  ADP_LOG(INFO) << "[GEOP] Start analyze input tensor.";
  NpuGetNextOutputInfo *output_info = static_cast<NpuGetNextOutputInfo *>(tensor_ptr);
  std::vector<int64> tmp_dims;
  for (int64_t dim : output_info->dims_) {
    tmp_dims.push_back(dim);
  }
  TensorShape input_shape(tmp_dims);
  if (need_collect_shapes) {
    input_shapes.push_back(input_shape.DebugString());
  }
  ge::Shape ge_shape(output_info->dims_);
  ge::TensorDesc ge_tensor_desc(ge_shape);
  ge_tensor_desc.SetOriginShape(ge_shape);
  ge_tensor_desc.SetDataType(type);
  ge_tensor_desc.SetPlacement(output_info->placement_);
  input.SetTensorDesc(ge_tensor_desc);

  uint8_t *data = output_info->data_.release();
  input.SetData(data, output_info->output_size_, output_info->data_.get_deleter());
  ADP_LOG(INFO) << "[GEOP] Get input shape: " << input_shape.DebugString()
                << ", input placement: " << output_info->placement_ << ", input length: " << output_info->output_size_
                << ", input data addr: " << reinterpret_cast<uintptr_t>(data);
}

Status GeOp::AnalyzeStringInput(ge::Tensor &input, const std::vector<std::string> &string_vector) const {
  const size_t count = string_vector.size();
  uint64_t total_size = 0U;
  for (uint64_t i = 0U; i < count; i++) {
    total_size += (string_vector[i].size() + sizeof(ge::StringHead) + 1U);
  }

  std::unique_ptr<char[]> addr(new (std::nothrow) char[total_size]());
  REQUIRES_NOT_NULL(addr);
  ge::StringHead *string_head = ge::PtrToPtr<char, ge::StringHead>(addr.get());
  char *data_addr = addr.get() + count * sizeof(ge::StringHead);
  int64_t offset = static_cast<int64_t>(count * sizeof(ge::StringHead));
  for (uint64_t i = 0U; i < count; ++i) {
    string_head[i].addr = offset;
    const string &str = string_vector[i];
    string_head[i].len = static_cast<int64_t>(str.size());
    size_t str_size = str.size();
    const char *string_addr = str.c_str();
    while (str_size >= SECUREC_MEM_MAX_LEN) {
      const auto ret = memcpy_s(data_addr, SECUREC_MEM_MAX_LEN, string_addr, SECUREC_MEM_MAX_LEN);
      NPU_REQUIRES(ret == EOK, errors::Internal("call memcpy_s failed, ret: ", ret));
      str_size -= SECUREC_MEM_MAX_LEN;
      offset += SECUREC_MEM_MAX_LEN;
      data_addr += SECUREC_MEM_MAX_LEN;
      string_addr += SECUREC_MEM_MAX_LEN;
    }
    auto remain_size = ((total_size - offset) > SECUREC_MEM_MAX_LEN) ? SECUREC_MEM_MAX_LEN : (total_size - offset);
    const auto ret = memcpy_s(data_addr, remain_size, string_addr, str_size + 1U);
    NPU_REQUIRES(ret == EOK, errors::Internal("call memcpy_s failed, ret:", ret));
    data_addr += (str_size + 1U);
    offset += (static_cast<int64_t>(str_size) + 1);
  }
  ADP_LOG(INFO) << "[GEOP] String input total size " << total_size << ", elements num: " << count;
  input.SetData(ge::PtrToPtr<char, const uint8_t>(addr.get()), total_size);
  return Status::OK();
}

Status GeOp::GraphInputConvertToConst(OpKernelContext *ctx) {
  mutex_lock lock{graph_handler_.graph_mu};
  if (is_input_convert_) {
    return Status::OK();
  }
  ADP_LOG(INFO) << "Begin to convert input to const.";
  is_input_convert_ = true;
  NPU_REQUIRES(ctx->function_library() != nullptr,
               errors::Internal("Function:", function_.name(), " ctx function is nullptr"));
  FunctionLibraryDefinition *func_lib_def =
      const_cast<FunctionLibraryDefinition *>(ctx->function_library()->GetFunctionLibraryDefinition());
  NPU_REQUIRES(func_lib_def != nullptr,
               errors::Internal("Function:", function_.name(), " func_lib_def def is nullptr"));
  const FunctionDef *function_def = func_lib_def->Find(function_.name());
  NPU_REQUIRES(function_def != nullptr, errors::Internal("Function:", function_.name(), " fdef is nullptr"));

  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(InferShapeUtil::GetSubGraphFromFunctionDef(*func_lib_def, *function_def, &graph));
  for (Node *node : graph.nodes()) {
    if (node->type_string() != "_Arg") {
      continue;
    }

    bool check_value = false;
    for (auto out : node->out_edges()) {
      if (out->dst()->attrs().Find(ATTR_NAME_CONST_INPUT_NAME) == nullptr) {
        continue;
      }
      std::vector<std::string> attr_consts;
      TF_RETURN_IF_ERROR(GetNodeAttr(out->dst()->attrs(), ATTR_NAME_CONST_INPUT_NAME, &attr_consts));
      if (attr_consts.at(out->dst_input()) != "") {
        check_value = true;
      }
    }

    if (check_value) {
      int32_t index = 0;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &index));
      Tensor tensor(ctx->input(index));
      std::string const_input_name = "Const" + std::to_string(index);
      Node *const_node = nullptr;
      TF_CHECK_OK(NodeBuilder(const_input_name, "Const")
                      .Device(node->def().device())
                      .Attr("dtype", tensor.dtype())
                      .Attr("value", tensor)
                      .Finalize(&graph, &const_node));
      for (auto out_edge : node->out_edges()) {
        REQUIRES_NOT_NULL(out_edge);
        graph.AddEdge(const_node, out_edge->src_output(), out_edge->dst(), out_edge->dst_input());
      }
      graph.RemoveNode(node);
      remove_index_.push_back(std::make_pair(tensor, index));
    }
  }

  if (remove_index_.size() == 0) {
    ADP_LOG(INFO) << "[GEOP] Return for don't have const input.";
    return Status::OK();
  }

  // refresh the index attr for _Arg
  std::vector<std::pair<Node *, int32_t>> input_infos;
  for (Node *node : graph.nodes()) {
    if (node->type_string() != "_Arg") {
      continue;
    }
    int32_t index = 0;
    (void)GetNodeAttr(node->attrs(), "index", &index);
    input_infos.emplace_back(std::make_pair(node, index));
  }
  std::sort(input_infos.begin(), input_infos.end(), CmpNodeIndex);
  int32_t input_index = 0;
  for (auto &input_info : input_infos) {
    input_info.first->AddAttr("index", input_index);
    input_index++;
  }

  FunctionDefLibrary fdeflib;
  FunctionDef *const_fdef = fdeflib.add_function();
  NPU_REQUIRES_OK(GraphToFunctionDef(graph, function_.name(), const_fdef));
  NPU_REQUIRES_OK(func_lib_def->RemoveFunction(function_.name()));
  NPU_REQUIRES_OK(func_lib_def->AddFunctionDef(*const_fdef));
  ADP_LOG(INFO) << "[GEOP] Convert input to const success.";
  return Status::OK();
}

Status GeOp::GraphCheckInputEqualConstOp(Tensor &tensor, int32_t index, bool &is_equal) {
  mutex_lock lock{graph_handler_.graph_mu};
  if (remove_index_.size() == 0) {
    return Status::OK();
  }

  for (auto it : remove_index_) {
    if (it.second != index) {
      continue;
    }
    char *tensor_const = ge::PtrToPtr<void, char>(DMAHelper::base(&it.first));
    char *tensor_input = ge::PtrToPtr<void, char>(DMAHelper::base(&tensor));
    is_equal = ((it.first.TotalBytes() == tensor.TotalBytes()) &&
                (memcmp(tensor_const, tensor_input, tensor.TotalBytes()) == 0));
    if (!is_equal) {
      return errors::Internal("Const input not equal with the input tensor value.");
    }
  }
  ADP_LOG(INFO) << "[GEOP] The input with const flag equal const op value.";
  return Status::OK();
}

Status GeOp::BuildInputTensorInfo(OpKernelContext *const ctx, std::vector<Tensor> &input_vec,
                                  std::vector<std::string> &input_shapes, std::vector<ge::Tensor> &inputs) {
  // ctx is not nullptr
  int32_t num_inputs = ctx->num_inputs();
  // populate inputs
  inputs.reserve(num_inputs);
  input_vec.reserve(num_inputs);
  const bool need_collect_shapes = (!IsDynamicConfig() && IsLazyCompile());
  for (int32_t i = 0; i < num_inputs; i++) {
    Tensor tensor(ctx->input(i));
    bool is_equal = false;
    if (GraphCheckInputEqualConstOp(tensor, i, is_equal) != Status::OK()) {
      return errors::Internal("Const op value not equal with tensor :", i);
    } else if (is_equal) {
      continue;
    }
    DataType data_type = tensor.dtype();
    auto tensor_ptr = static_cast<void *>(const_cast<char *>(tensor.tensor_data().data()));
    auto tensor_size = tensor.tensor_data().size();

    ge::DataType type;
    (void)GeApiWrapper_GetGeDataTypeByTFType(static_cast<uint32_t>(data_type), type);
    if (type == ge::DT_UNDEFINED) {
      ADP_LOG(ERROR) << "[GEOP] No Supported datatype : " << data_type;
      LOG(ERROR) << "[GEOP] No Supported datatype : " << data_type;
      return errors::InvalidArgument("No Supported datatype : ", data_type);
    }
    ge::Tensor input;
    if (is_dynamic_getnext_ == "1" && (placeholder_index_.find(std::to_string(i)) == std::string::npos)) {
      REQUIRES_NOT_NULL(tensor_ptr);
      AnalyzeInputDesc(need_collect_shapes, tensor_ptr, input, type, input_shapes);
    } else {
      std::vector<int64_t> dims;
      for (uint32_t dim : tensor.shape().dim_sizes()) {
        dims.push_back(static_cast<int64_t>(dim));
      }
      ge::Shape ge_shape(dims);
      ge::TensorDesc ge_tensor_desc(ge_shape);
      ge_tensor_desc.SetDataType(type);
      ge_tensor_desc.SetOriginShape(ge_shape);
      input.SetTensorDesc(ge_tensor_desc);
      if (type == ge::DT_STRING) {
        const uint64_t count = static_cast<uint64_t>(tensor.NumElements());
        std::vector<std::string> string_vector;
        for (uint64_t i = 0UL; i < count; i++) {
          string_vector.emplace_back(tensor.flat<tstring>()(i));
        }
        ADP_LOG(INFO) << "[GEOP] Analyze string input: " << i << ", element num: " << count;
        if (AnalyzeStringInput(input, string_vector) != Status::OK()) {
          return errors::Internal("The input string data analyze failed.");
        }
      } else {
        input.SetData(static_cast<uint8_t *>(tensor_ptr), tensor_size, [](uint8_t *) {});
      }
      if (need_collect_shapes) {
        input_shapes.push_back(tensor.shape().DebugString());
      }
    }
    inputs.push_back(input);
    input_vec.push_back(tensor);
  }
  return Status::OK();
}

Status GeOp::BuildOutTensorInfo(OpKernelContext *ctx) {
  int num_outputs = ctx->num_outputs();
  // populate outputs
  for (int i = 0; i < num_outputs; i++) {
    TensorShape out_shape = outputs_shape_.at(i);
    Tensor *tensor = nullptr;
    TF_RETURN_IF_ERROR(ctx->allocate_output(i, out_shape, &tensor));
  }
  return Status::OK();
}

// For each NodeDef, Create Input&Output Desc(shape,format,dataType)
Status GeOp::GenerateDesc(Node *&node) {
  REQUIRES_NOT_NULL(node);
  NodeDef &node_def = const_cast<NodeDef &>(node->def());
  const OpDef &op_def = node->op_def();

  std::string format = this->data_format_;
  int32_t domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_RESERVED;
  TF_RETURN_IF_ERROR(this->DomiFormatFromString(format, domi_format));

  // Get signature(dataType) from the OpDef & NodeDef
  DataTypeVector inputs;
  DataTypeVector outputs;
  TF_RETURN_IF_ERROR(tensorflow::InOutTypesForNode(node_def, op_def, &inputs, &outputs));

  int32_t num;
  Node *in_node = nullptr;
  const Edge *in_edge = nullptr;

  if (inputs.size() > INT_MAX) {
    return errors::InvalidArgument("inputs size should be less than INT_MAX.");
  }

  // Create input Desc
  int32_t inputs_size = static_cast<int32_t>(inputs.size());
  if (inputs_size > 0) {
    AttrValue input_tensor_descs;
    AttrValue input_tensor_descs_s;
    num = 0;
    for (; num < inputs_size;) {
      node->input_node(num, &in_node);
      node->input_edge(num, &in_edge);
      REQUIRES_NOT_NULL(in_node);
      REQUIRES_NOT_NULL(in_edge);
      int32_t src_output = in_edge->src_output();
      if (in_node->def().attr().find(OUTPUT_DESC) != in_node->def().attr().end()) {
        const AttrValue_ListValue &attr_list = in_node->def().attr().at(OUTPUT_DESC).list();
        if (attr_list.func_size() > src_output) {
          NameAttrList desc_attr = in_node->def().attr().at(OUTPUT_DESC).list().func(src_output);
          *(input_tensor_descs.mutable_list()->add_func()) = desc_attr;
        } else {
          NameAttrList name_attr_list;
          name_attr_list.set_name(std::to_string(0));
          AttrValue attr_format_value;
          attr_format_value.set_i(static_cast<int64_t>(domi_format));
          name_attr_list.mutable_attr()->insert({SERIALIZE_FORMAT, attr_format_value});
          AttrValue attr_datatype_value;
          attr_datatype_value.set_i(static_cast<int64_t>(inputs[num]));
          name_attr_list.mutable_attr()->insert({SERIALIZE_DATATYPE, attr_datatype_value});
          AttrValue attr_shape_value;
          attr_shape_value.set_type(DT_INT32);
          name_attr_list.mutable_attr()->insert({SERIALIZE_SHAPE, attr_shape_value});
          *(input_tensor_descs.mutable_list()->add_func()) = name_attr_list;
        }
      } else {
        ADP_LOG(INFO) << "[GEOP] no OUTPUT_DESC: " << node->name() << " <-- " << in_node->name();
        if (num > 0 && node->type_string() == "Merge" && in_node->type_string() == "NextIteration") {
          node->input_node(num - 1, &in_node);
          node->input_edge(num - 1, &in_edge);
          REQUIRES_NOT_NULL(in_node);
          REQUIRES_NOT_NULL(in_edge);
          int pre_src_output = in_edge->src_output();
          NameAttrList desc_attr = in_node->def().attr().at(OUTPUT_DESC).list().func(pre_src_output);
          *(input_tensor_descs.mutable_list()->add_func()) = desc_attr;
        }
      }
      num++;
    }
    REQUIRES_NOT_NULL(node_def.mutable_attr());
    node_def.mutable_attr()->insert({INPUT_DESC, input_tensor_descs});
  }

  // Create output Desc
  if (outputs.size() > 0) {
    // Get infershape
    const std::string key_shape = tensorflow::KEY_SHAPE;
    AttrValue shape_value;
    const auto &it = node_def.attr().find(key_shape);
    if (it == node_def.attr().end()) {  // no find
      ADP_LOG(WARNING) << "[GEOP] There is no shape of node : " << node_def.name();
    } else {
      shape_value = node_def.attr().at(key_shape);
      uint32_t shape_size = static_cast<uint32_t>(shape_value.list().shape_size());
      if (shape_size != outputs.size()) {
        ADP_LOG(ERROR) << "[GEOP] size not equal, shape_size : " << shape_size << " outputs size:" << outputs.size();
        LOG(ERROR) << "[GEOP] size not equal, shape_size : " << shape_size << " outputs size:" << outputs.size();
        shape_value.clear_list();
      }
    }
    // Create output Desc
    AttrValue output_tensor_descs;
    AttrValue output_tensor_descs_s;
    int32_t i = 0;
    num = 0;
    for (DataType data_type : outputs) {
      string desc_string_s;
      AttrValue attr_format_value;
      attr_format_value.set_i(static_cast<int64_t>(domi_format));
      AttrValue attr_datatype_value;
      attr_datatype_value.set_i(static_cast<int64_t>(data_type));

      // shape
      AttrValue attr_shape_value;
      attr_shape_value.set_type(DT_INT32);
      if (shape_value.has_list()) {
        TensorShapeProto shape_proto = shape_value.list().shape(num);
        if (shape_proto.unknown_rank()) {
          attr_shape_value.mutable_list()->add_i(-2);
          ADP_LOG(INFO) << "Node: " << node_def.name() << " set unknown rank";
        }
        for (int j = 0; j < shape_proto.dim_size(); j++) {
          attr_shape_value.mutable_list()->add_i(shape_proto.dim(j).size());
          ADP_LOG(INFO) << "Node: " << node_def.name() << " set dim[" << j << "] = " << shape_proto.dim(j).size();
        }
      } else {
        attr_shape_value.mutable_list()->add_i(-2);
        ADP_LOG(INFO) << "Node: " << node_def.name() << " set unknown rank";
      }

      NameAttrList name_attr_list;
      name_attr_list.set_name(std::to_string(i));
      REQUIRES_NOT_NULL(name_attr_list.mutable_attr());
      name_attr_list.mutable_attr()->insert({SERIALIZE_FORMAT, attr_format_value});
      name_attr_list.mutable_attr()->insert({SERIALIZE_DATATYPE, attr_datatype_value});
      name_attr_list.mutable_attr()->insert({SERIALIZE_SHAPE, attr_shape_value});
      REQUIRES_NOT_NULL(output_tensor_descs.mutable_list());
      *(output_tensor_descs.mutable_list()->add_func()) = name_attr_list;

      num++;
      i++;
    }
    node_def.mutable_attr()->erase(key_shape);
    node_def.mutable_attr()->insert({OUTPUT_DESC, output_tensor_descs});
  }
  string op_def_string;
  op_def.SerializeToString(&op_def_string);

  tensorflow::AttrValue value;
  value.set_s(op_def_string);
  node_def.mutable_attr()->insert({"op_def", value});
  return tensorflow::Status::OK();
}

Status GeOp::DomiFormatFromString(std::string format, int32_t &domi_format) const {
  if (format == "NCHW") {
    domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_NCHW;
    return Status::OK();
  } else if (format == "NHWC") {
    domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_NHWC;
    return Status::OK();
  } else if (format == "NC1HWC0") {
    domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_NC1HWC0;
    return Status::OK();
  } else if (format == "NDHWC") {
    domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_NDHWC;
    return Status::OK();
  } else if (format == "NCDHW") {
    domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_NCDHW;
    return Status::OK();
  } else if (format == "DHWCN") {
    domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_DHWCN;
    return Status::OK();
  } else if (format == "DHWNC") {
    domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_DHWNC;
    return Status::OK();
  } else if (format == "FRACTALZ") {
    domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_FRACTAL_Z;
    return Status::OK();
  } else if (format == "ND") {
    domi_format = domi::domiTensorFormat_t::DOMI_TENSOR_ND;
    return Status::OK();
  }
  return errors::Internal("DomiFormatFromString, not supported format, format = ", format);
}
void GeOp::InitAoeFlag() {
  is_aoe_ = (!init_options_["ge.jobType"].empty()) && (!init_options_["ge.tuningPath"].empty());
}
}  // namespace tensorflow

namespace tensorflow {
mutex GeOp::mu_(LINKER_INITIALIZED);
bool GeOp::tuned_initialize_flag_(false);

const std::string GeOp::INPUT_DESC = "input_tensor_desc";
const std::string GeOp::OUTPUT_DESC = "output_tensor_desc";
const std::string GeOp::SERIALIZE_FORMAT = "serialize_format";
const std::string GeOp::SERIALIZE_DATATYPE = "serialize_datatype";
const std::string GeOp::SERIALIZE_SHAPE = "serialize_shape";
const std::string GeOp::SubGraph = "SubGraph";
std::unordered_map<std::string, uint32_t> GeOp::session_and_graph_id_map_;

REGISTER_KERNEL_BUILDER(Name("GeOp").Device(DEVICE_CPU), GeOp);
}  // namespace tensorflow
