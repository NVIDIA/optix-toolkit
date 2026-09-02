// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <mi/mdl_sdk.h>

namespace demandPbrtScene {

using BackendApiHandle              = mi::base::Handle<mi::neuraylib::IMdl_backend_api>;
using BackendHandle                 = mi::base::Handle<mi::neuraylib::IMdl_backend>;
using BsdfMeasurementHandle         = mi::base::Handle<mi::neuraylib::IBsdf_measurement>;
using ColorValueHandle              = mi::base::Handle<mi::neuraylib::IValue_color>;
using CompiledMaterialHandle        = mi::base::Handle<mi::neuraylib::ICompiled_material>;
using ConstColorValueHandle         = mi::base::Handle<const mi::neuraylib::IValue_color>;
using ConstExpressionConstantHandle = mi::base::Handle<const mi::neuraylib::IExpression_constant>;
using ConstExpressionHandle         = mi::base::Handle<const mi::neuraylib::IExpression>;
using ConstFloatValueHandle         = mi::base::Handle<const mi::neuraylib::IValue_float>;
using ConstStringHandle             = mi::base::Handle<const mi::IString>;
using DatabaseHandle                = mi::base::Handle<mi::neuraylib::IDatabase>;
using ExecutionContextHandle        = mi::base::Handle<mi::neuraylib::IMdl_execution_context>;
using ExpressionConstantHandle      = mi::base::Handle<mi::neuraylib::IExpression_constant>;
using ExpressionFactoryHandle       = mi::base::Handle<mi::neuraylib::IExpression_factory>;
using FloatValueHandle              = mi::base::Handle<mi::neuraylib::IValue_float>;
using FunctionCallHandle            = mi::base::Handle<mi::neuraylib::IFunction_call>;
using FunctionDefinitionHandle      = mi::base::Handle<const mi::neuraylib::IFunction_definition>;
using MaterialInstanceHandle        = mi::base::Handle<mi::neuraylib::IMaterial_instance>;
using MdlFactoryHandle              = mi::base::Handle<mi::neuraylib::IMdl_factory>;
using MdlImpexpApiHandle            = mi::base::Handle<mi::neuraylib::IMdl_impexp_api>;
using MessageHandle                 = mi::base::Handle<const mi::neuraylib::IMessage>;
using ModuleHandle                  = mi::base::Handle<const mi::neuraylib::IModule>;
using NeurayHandle                  = mi::base::Handle<mi::neuraylib::INeuray>;
using ScopeHandle                   = mi::base::Handle<mi::neuraylib::IScope>;
using TargetArgumentBlockHandle     = mi::base::Handle<const mi::neuraylib::ITarget_argument_block>;
using TargetCodeHandle              = mi::base::Handle<const mi::neuraylib::ITarget_code>;
using TargetValueLayoutHandle       = mi::base::Handle<const mi::neuraylib::ITarget_value_layout>;
using TransactionHandle             = mi::base::Handle<mi::neuraylib::ITransaction>;
using TypeFactoryHandle             = mi::base::Handle<mi::neuraylib::IType_factory>;
using TypeHandle                    = mi::base::Handle<const mi::neuraylib::IType>;
using ValueFactoryHandle            = mi::base::Handle<mi::neuraylib::IValue_factory>;
using VersionHandle                 = mi::base::Handle<const mi::neuraylib::IVersion>;

}  // namespace demandPbrtScene
