/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <vector>

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"
#include "gtest/gtest.h"

namespace tensorflow {
namespace {

class DummyDevice : public DeviceBase {
 public:
  DummyDevice() : DeviceBase(Env::Default()) {}

  bool RequiresRecordingAccessedTensors() const override {
    return false;
  }

  Allocator *GetAllocator(AllocatorAttributes) override {
    return cpu_allocator();
  }
};

TEST(UniqueParallelOpsTest, PreservesFirstAppearanceOrder) {
  NodeDef node_def;
  ASSERT_TRUE(NodeDefBuilder("unique", "Unique")
                  .Input("input", 0, DT_INT32)
                  .Attr("T", DT_INT32)
                  .Attr("out_idx", DT_INT32)
                  .Attr("_kernel", "parallel")
                  .Finalize(&node_def)
                  .ok());

  DummyDevice device;
  Status status;
  std::unique_ptr<OpKernel> op(
      CreateOpKernel(DEVICE_CPU, &device, cpu_allocator(), node_def, TF_GRAPH_DEF_VERSION, &status));
  ASSERT_TRUE(status.ok()) << status.ToString();

  Tensor input(DT_INT32, TensorShape({3}));
  input.vec<int32>()(0) = 2;
  input.vec<int32>()(1) = 1;
  input.vec<int32>()(2) = 2;
  gtl::InlinedVector<TensorValue, 4> inputs{TensorValue(&input)};
  OpKernelContext::Params params;
  params.device = &device;
  params.op_kernel = op.get();
  params.inputs = &inputs;
  std::vector<AllocatorAttributes> output_attrs(op->num_outputs());
  params.output_attr_array = output_attrs.data();
  OpKernelContext context(&params);

  op->Compute(&context);
  ASSERT_TRUE(context.status().ok()) << context.status().ToString();
  const Tensor &output = *context.mutable_output(0);
  const Tensor &indices = *context.mutable_output(1);
  ASSERT_EQ(output.NumElements(), 2);
  EXPECT_EQ(output.vec<int32>()(0), 2);
  EXPECT_EQ(output.vec<int32>()(1), 1);
  EXPECT_EQ(indices.vec<int32>()(0), 0);
  EXPECT_EQ(indices.vec<int32>()(1), 1);
  EXPECT_EQ(indices.vec<int32>()(2), 0);
}

}  // namespace
}  // namespace tensorflow
