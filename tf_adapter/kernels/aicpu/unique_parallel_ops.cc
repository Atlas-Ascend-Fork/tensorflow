/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <atomic>
#include <numeric>
#include <unordered_map>
#include <vector>
#include "tf_adapter/common/common.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#define EIGEN_USE_THREADS
#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_THREAD_POOL_H

namespace tensorflow {
template <typename T, typename TIndex>
class UniqueParallelOp : public OpKernel {
 public:
  explicit UniqueParallelOp(OpKernelConstruction *context) : OpKernel(context) {}
  ~UniqueParallelOp() override = default;
  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    OP_REQUIRES(context, input_tensor.NumElements() <= std::numeric_limits<int32>::max(),
                errors::InvalidArgument("unique does not support input tensors larger than ",
                                        std::numeric_limits<int32>::max(), " elements"));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor.shape()),
                errors::InvalidArgument("unique expects a 1D vector."));
    OP_REQUIRES(context, (input_tensor.dtype() == DT_INT32 || input_tensor.dtype() == DT_INT64),
                errors::InvalidArgument("input tensor should be int32 or int64, but got ",
                                        DataTypeString(input_tensor.dtype())));
    auto input_vec = input_tensor.vec<T>();
    int64 total = static_cast<int64>(input_vec.size());
    Tensor *index_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({input_vec.size()}), &index_tensor));
    auto index_vec = index_tensor->vec<TIndex>();

    const int64 CPU_NUMS = 16;
    std::atomic<TIndex> count_num(0);
    Tensor temp_output(input_tensor.dtype(), TensorShape({total}));
    auto temp_output_flat = temp_output.flat<T>();
    tensorflow::thread::ThreadPool thread_work(context->env(), "unique_parallel", CPU_NUMS);
    std::function<void(int64, int)> shards = [&](int64 total, int cur) {
      std::unordered_map<T, TIndex> unique_map;
      for (TIndex i = 0, j = 0; i < total; i++) {
        unsigned long long temp_input = input_vec(i);
        unsigned long long temp_cpu = CPU_NUMS - 1;
        unsigned long long temp_cur = cur;
        if ((temp_input & temp_cpu) == temp_cur) {
          if (unique_map.find(input_vec(i)) == unique_map.end()) {
            j = count_num++;
            unique_map[input_vec(i)] = j;
            temp_output_flat(j) = input_vec(i);
          }
          index_vec(i) = unique_map[input_vec(i)];
        }
      }
    };
    ParallelFor(thread_work, total, CPU_NUMS, shards);
    // TensorFlow Unique orders y by first appearance, while the parallel shards assign slots by completion order.
    const TIndex unique_count = count_num.load();
    RestoreFirstAppearanceOrder(temp_output, *index_tensor, unique_count);
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({unique_count}), &output_tensor));
    *output_tensor = temp_output.Slice(0, unique_count);
  }

 private:
  void RestoreFirstAppearanceOrder(Tensor &temp_output, Tensor &index_tensor, TIndex unique_count) {
    auto temp_output_flat = temp_output.flat<T>();
    auto index_vec = index_tensor.vec<TIndex>();
    const int64 total = static_cast<int64>(index_vec.size());
    std::vector<int64> first_positions(static_cast<size_t>(unique_count), total);
    for (int64 i = 0; i < total; i++) {
      const TIndex old_index = index_vec(i);
      if (first_positions[static_cast<size_t>(old_index)] == total) {
        first_positions[static_cast<size_t>(old_index)] = i;
      }
    }
    std::vector<TIndex> order(static_cast<size_t>(unique_count));
    std::iota(order.begin(), order.end(), static_cast<TIndex>(0));
    std::sort(order.begin(), order.end(), [&first_positions](TIndex lhs, TIndex rhs) {
      return first_positions[static_cast<size_t>(lhs)] < first_positions[static_cast<size_t>(rhs)];
    });
    std::vector<TIndex> old_to_new(static_cast<size_t>(unique_count));
    std::vector<T> ordered_output(static_cast<size_t>(unique_count));
    for (TIndex new_index = 0; new_index < unique_count; new_index++) {
      const TIndex old_index = order[static_cast<size_t>(new_index)];
      old_to_new[static_cast<size_t>(old_index)] = new_index;
      ordered_output[static_cast<size_t>(new_index)] = temp_output_flat(old_index);
    }
    for (TIndex new_index = 0; new_index < unique_count; new_index++) {
      temp_output_flat(new_index) = ordered_output[static_cast<size_t>(new_index)];
    }
    for (int64 i = 0; i < total; i++) {
      index_vec(i) = old_to_new[static_cast<size_t>(index_vec(i))];
    }
  }

  void ParallelFor(tensorflow::thread::ThreadPool &thread_work, int64 total, const int cpu_nums,
                   std::function<void(int64, int)> &fn) {
    CHECK_GE(total, 0);
    CHECK_EQ(total, static_cast<int64>(static_cast<Eigen::Index>(total)));
    if (total <= 1 || cpu_nums == 1) {
      fn(total, 0);
      return;
    }
    Eigen::Barrier barrier(static_cast<unsigned int>(cpu_nums));
    std::function<void(int64, int)> handle_range;
    handle_range = [=, &barrier, &fn](int64 total, int cur) {
      fn(total, cur);
      barrier.Notify();
    };
    for (int i = 0; i < cpu_nums; i++) {
      thread_work.Schedule([=, &handle_range]() { handle_range(total, i); });
    }
    barrier.Wait();
  }
};

/*lint -e665*/
#define REGISTER_UNIQUE_PARALLEL(type)                                                                                \
  REGISTER_KERNEL_BUILDER(                                                                                            \
      Name("Unique").Device(DEVICE_CPU).TypeConstraint<type>("T").TypeConstraint<int32>("out_idx").Label("parallel"), \
      UniqueParallelOp<type, int32>);                                                                                 \
  REGISTER_KERNEL_BUILDER(                                                                                            \
      Name("Unique").Device(DEVICE_CPU).TypeConstraint<type>("T").TypeConstraint<int64>("out_idx").Label("parallel"), \
      UniqueParallelOp<type, int64>);
TF_CALL_INTEGRAL_TYPES(REGISTER_UNIQUE_PARALLEL);
#undef REGISTER_UNIQUE_PARALLEL
/*lint +e665*/
}  // namespace tensorflow
