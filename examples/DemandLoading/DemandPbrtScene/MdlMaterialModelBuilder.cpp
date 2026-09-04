// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/MdlMaterialModelBuilder.h"

#ifdef OTK_USE_MDL

#include "DemandPbrtScene/MdlKeyBuilder.h"
#include "DemandPbrtScene/MdlParameterBinder.h"
#include "DemandPbrtScene/MdlTextureGraphGenerator.h"
#include "DemandPbrtScene/PbrtMaterialKind.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace demandPbrtScene {

void appendUnsupportedReason( GeneratedMdlSource& result, const std::string& reason )
{
    if( std::find( result.unsupportedReasons.begin(), result.unsupportedReasons.end(), reason )
        == result.unsupportedReasons.end() )
    {
        result.unsupportedReasons.push_back( reason );
    }
}

namespace {

std::string stableHash( const std::string& text )
{
    std::uint64_t hash{ 14695981039346656037ULL };
    for( std::string::const_iterator it = text.begin(); it != text.end(); ++it )
    {
        hash ^= static_cast<unsigned char>( *it );
        hash *= 1099511628211ULL;
    }

    std::ostringstream out;
    out << std::hex << std::setfill( '0' ) << std::setw( 16 ) << hash;
    return out.str();
}

struct MdlMaterialParameter
{
    std::string type;
    std::string name;
    std::string defaultValue;
};

struct MdlMaterialModel
{
    std::vector<MdlMaterialParameter> parameters;
    std::vector<std::string>          comments;
    std::string                       helperDefinitions;
    std::string                       body;
};

struct PbrtMaterialGapPolicy
{
    PbrtMaterialKind kind;
    std::string      policy;
    std::string      coverageReason;
};

void appendMaterialParameter( MdlMaterialModel& model, const std::string& type, const std::string& name, const std::string& defaultValue )
{
    model.parameters.push_back( MdlMaterialParameter{ type, name, defaultValue } );
}

const PbrtMaterialGapPolicy* explicitMaterialGapPolicy( PbrtMaterialKind kind )
{
    static const PbrtMaterialGapPolicy policies[] = {
        { PbrtMaterialKind::FOURIER, "unsupported with visible fallback",
          "PBRT Fourier tables are data-driven BSDF resources found in the corpus; DemandPbrtScene preserves the "
          "resource metadata but does not yet evaluate the Fourier table on the GPU" },
        { PbrtMaterialKind::HAIR, "unsupported with visible fallback",
          "low-frequency PBRT corpus material; no current target scene or reference fixture requires approximation" },
        { PbrtMaterialKind::MEASURED, "unsupported with visible fallback",
          "PBRT parity completeness gap; current corpus sample did not find a target scene requiring support" },
    };

    for( const PbrtMaterialGapPolicy& policy : policies )
    {
        if( policy.kind == kind )
        {
            return &policy;
        }
    }
    return nullptr;
}

bool hasFourierBsdfFile( const otk::pbrt::PbrtMaterial& material )
{
    return !material.params.FindOneString( "bsdffile", std::string{} ).empty();
}

void appendRoughnessGapComment( MdlMaterialModel& model )
{
    model.comments.push_back( "pbrt material gap: PBRT-exact roughness/remapping behavior is approximated" );
}

std::string mdlParameterList( const std::vector<MdlMaterialParameter>& parameters )
{
    if( parameters.empty() )
    {
        return "()";
    }

    std::ostringstream out;
    out << "(\n";
    for( std::vector<MdlMaterialParameter>::const_iterator it = parameters.begin(); it != parameters.end(); ++it )
    {
        out << "    " << it->type << " " << it->name << " = " << it->defaultValue;
        if( it + 1 != parameters.end() )
        {
            out << ",";
        }
        out << "\n";
    }
    out << ")";
    return out.str();
}

std::string materialTextureCommentExpression( MdlTextureGraphGenerator& textureGraph,
                                              const ::pbrt::ParamSet&   params,
                                              const std::string&        paramName,
                                              const std::string&        preferredValueType )
{
    if( params.FindTexture( paramName ).empty() )
    {
        return "none";
    }
    return textureGraph.materialColorExpression( params, paramName, preferredValueType, "none" );
}

std::string materialBumpmapExpression( MdlTextureGraphGenerator& textureGraph, const ::pbrt::ParamSet& params )
{
    if( params.FindTexture( "bumpmap" ).empty() )
    {
        return "none";
    }
    return textureGraph.materialFloatExpression( params, "bumpmap", "float", "0.0" );
}

bool hasBumpmapExpression( const std::string& bumpmap )
{
    return bumpmap != "none";
}

void appendBumpmapCommentsAndHelpers( MdlMaterialModel& model, const std::string& bumpmap )
{
    model.comments.push_back( "pbrt material input bumpmap: " + bumpmap );
    if( !hasBumpmapExpression( bumpmap ) )
    {
        return;
    }

    model.comments.push_back( "pbrt material implementation: bumpmap is evaluated with runtime finite differences" );
}

std::string materialGeometryExpression( const std::string& cutoutOpacity, const std::string& bumpmap )
{
    (void)bumpmap;
    if( cutoutOpacity.empty() )
    {
        return std::string{};
    }

    std::ostringstream out;
    out << "    geometry: material_geometry(\n";
    out << "        cutout_opacity: " << cutoutOpacity << "\n";
    out << "    )\n";
    return out.str();
}

std::string namedMaterialColorExpression( MdlMaterialModel&                   model,
                                          MdlTextureGraphGenerator&           textureGraph,
                                          const otk::pbrt::PbrtNamedMaterial& material,
                                          unsigned int                        index,
                                          const std::string&                  paramName,
                                          const std::string&                  defaultValue )
{
    const std::string parameterName{ namedMaterialParameterName( index, paramName ) };
    appendMaterialParameter( model, "color", parameterName, defaultValue );
    return textureGraph.materialColorExpression( material.params, paramName, "color", parameterName );
}

std::string namedMaterialFloatExpression( MdlMaterialModel&                   model,
                                          const otk::pbrt::PbrtNamedMaterial& material,
                                          unsigned int                        index,
                                          const std::string&                  paramName,
                                          const std::string&                  defaultValue )
{
    const std::string parameterName{ namedMaterialParameterName( index, paramName ) };
    appendMaterialParameter( model, "float", parameterName, defaultValue );
    if( !material.params.FindTexture( paramName ).empty() )
    {
        return defaultValue;
    }
    return parameterName;
}

std::string namedMaterialMatteBsdfExpression( MdlMaterialModel&                   model,
                                              MdlTextureGraphGenerator&           textureGraph,
                                              const otk::pbrt::PbrtNamedMaterial& material,
                                              unsigned int                        index )
{
    const std::string kd{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kd", "color(0.8, 0.8, 0.8)" ) };
    const std::string sigma{ namedMaterialFloatExpression( model, material, index, "sigma", "0.0" ) };
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kd: " + kd );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input sigma: " + sigma );
    return "::df::diffuse_reflection_bsdf(\n"
           "                        tint: "
           + kd
           + ",\n"
             "                        roughness: pbrt_mix_matte_sigma_roughness("
           + sigma + "))";
}

std::string namedMaterialPlasticBsdfExpression( MdlMaterialModel&                   model,
                                                MdlTextureGraphGenerator&           textureGraph,
                                                const otk::pbrt::PbrtNamedMaterial& material,
                                                unsigned int                        index )
{
    const std::string kd{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kd", "color(0.8, 0.8, 0.8)" ) };
    const std::string ks{
        namedMaterialColorExpression( model, textureGraph, material, index, "Ks", "color(0.0, 0.0, 0.0)" ) };
    const std::string roughness{ namedMaterialFloatExpression( model, material, index, "roughness", "0.1" ) };
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kd: " + kd );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Ks: " + ks );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input roughness: " + roughness );
    return "::df::color_normalized_mix(\n"
           "                        components: ::df::color_bsdf_component[](\n"
           "                            ::df::color_bsdf_component(\n"
           "                                weight: "
           + kd
           + ",\n"
             "                                component: ::df::diffuse_reflection_bsdf(\n"
             "                                    tint: color(1.0, 1.0, 1.0))),\n"
             "                            ::df::color_bsdf_component(\n"
             "                                weight: "
           + ks
           + ",\n"
             "                                component: ::df::simple_glossy_bsdf(\n"
             "                                    roughness_u: "
           + roughness
           + ",\n"
             "                                    roughness_v: "
           + roughness
           + ",\n"
             "                                    tint: color(1.0, 1.0, 1.0),\n"
             "                                    mode: ::df::scatter_reflect))))";
}

std::string namedMaterialSubstrateBsdfExpression( MdlMaterialModel&                   model,
                                                  MdlTextureGraphGenerator&           textureGraph,
                                                  const otk::pbrt::PbrtNamedMaterial& material,
                                                  unsigned int                        index )
{
    const std::string kd{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kd", "color(0.5, 0.5, 0.5)" ) };
    const std::string ks{
        namedMaterialColorExpression( model, textureGraph, material, index, "Ks", "color(0.5, 0.5, 0.5)" ) };
    const std::string roughness{ namedMaterialFloatExpression( model, material, index, "roughness", "0.1" ) };
    const std::string uroughness{ namedMaterialFloatExpression( model, material, index, "uroughness", "-1.0" ) };
    const std::string vroughness{ namedMaterialFloatExpression( model, material, index, "vroughness", "-1.0" ) };
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kd: " + kd );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Ks: " + ks );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input roughness: " + roughness );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input uroughness: " + uroughness );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input vroughness: " + vroughness );
    return "::df::color_weighted_layer(\n"
           "                        weight: "
           + ks
           + ",\n"
             "                        layer: ::df::simple_glossy_bsdf(\n"
             "                            roughness_u: pbrt_mix_resolved_roughness("
           + roughness + ", " + uroughness
           + "),\n"
             "                            roughness_v: pbrt_mix_resolved_roughness("
           + roughness + ", " + vroughness
           + "),\n"
             "                            tint: color(1.0, 1.0, 1.0),\n"
             "                            mode: ::df::scatter_reflect),\n"
             "                        base: ::df::diffuse_reflection_bsdf(\n"
             "                            tint: "
           + kd + "))";
}

std::string namedMaterialUberBsdfExpression( MdlMaterialModel&                   model,
                                             MdlTextureGraphGenerator&           textureGraph,
                                             const otk::pbrt::PbrtNamedMaterial& material,
                                             unsigned int                        index )
{
    const std::string kd{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kd", "color(0.8, 0.8, 0.8)" ) };
    const std::string ks{
        namedMaterialColorExpression( model, textureGraph, material, index, "Ks", "color(0.0, 0.0, 0.0)" ) };
    const std::string kr{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kr", "color(0.0, 0.0, 0.0)" ) };
    const std::string kt{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kt", "color(0.0, 0.0, 0.0)" ) };
    const std::string roughness{ namedMaterialFloatExpression( model, material, index, "roughness", "0.1" ) };
    const std::string uroughness{ namedMaterialFloatExpression( model, material, index, "uroughness", "-1.0" ) };
    const std::string vroughness{ namedMaterialFloatExpression( model, material, index, "vroughness", "-1.0" ) };
    const std::string alpha{ namedMaterialFloatExpression( model, material, index, "alpha", "1.0" ) };
    const std::string opacity{ namedMaterialFloatExpression( model, material, index, "opacity", "1.0" ) };
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kd: " + kd );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Ks: " + ks );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kr: " + kr );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kt: " + kt );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input roughness: " + roughness );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input uroughness: " + uroughness );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input vroughness: " + vroughness );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input opacity: " + opacity );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input alpha: " + alpha
                              + "; cutout does not compose through mix" );
    return std::string{ "::df::color_normalized_mix(\n"
                        "                        components: ::df::color_bsdf_component[](\n"
                        "                            ::df::color_bsdf_component(\n"
                        "                                weight: " }
           + "pbrt_mix_opacity_weight(" + opacity + ") * " + kd
           + ",\n"
             "                                component: ::df::diffuse_reflection_bsdf(\n"
             "                                    tint: color(1.0, 1.0, 1.0))),\n"
             "                            ::df::color_bsdf_component(\n"
             "                                weight: "
           + "pbrt_mix_opacity_weight(" + opacity + ") * " + ks
           + ",\n"
             "                                component: ::df::simple_glossy_bsdf(\n"
             "                                    roughness_u: pbrt_mix_resolved_roughness("
           + roughness + ", " + uroughness
           + "),\n"
             "                                    roughness_v: pbrt_mix_resolved_roughness("
           + roughness + ", " + vroughness
           + "),\n"
             "                                    tint: color(1.0, 1.0, 1.0),\n"
             "                                    mode: ::df::scatter_reflect)),\n"
             "                            ::df::color_bsdf_component(\n"
             "                                weight: "
           + "pbrt_mix_opacity_weight(" + opacity + ") * " + kr
           + ",\n"
             "                                component: ::df::specular_bsdf(\n"
             "                                    tint: color(1.0, 1.0, 1.0),\n"
             "                                    mode: ::df::scatter_reflect)),\n"
             "                            ::df::color_bsdf_component(\n"
             "                                weight: "
           + "pbrt_mix_opacity_weight(" + opacity + ") * " + kt
           + ",\n"
             "                                component: ::df::specular_bsdf(\n"
             "                                    tint: color(1.0, 1.0, 1.0),\n"
             "                                    mode: ::df::scatter_transmit)),\n"
             "                            ::df::color_bsdf_component(\n"
             "                                weight: pbrt_mix_transparency_weight("
           + opacity
           + "),\n"
             "                                component: ::df::specular_bsdf(\n"
             "                                    tint: color(1.0, 1.0, 1.0),\n"
             "                                    mode: ::df::scatter_transmit))))";
}

std::string namedMaterialMirrorBsdfExpression( MdlMaterialModel&                   model,
                                               MdlTextureGraphGenerator&           textureGraph,
                                               const otk::pbrt::PbrtNamedMaterial& material,
                                               unsigned int                        index )
{
    const std::string kr{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kr", "color(1.0, 1.0, 1.0)" ) };
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kr: " + kr );
    return "::df::specular_bsdf(\n"
           "                        tint: "
           + kr
           + ",\n"
             "                        mode: ::df::scatter_reflect)";
}

std::string namedMaterialGlassBsdfExpression( MdlMaterialModel&                   model,
                                              MdlTextureGraphGenerator&           textureGraph,
                                              const otk::pbrt::PbrtNamedMaterial& material,
                                              unsigned int                        index )
{
    const std::string kr{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kr", "color(1.0, 1.0, 1.0)" ) };
    const std::string kt{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kt", "color(1.0, 1.0, 1.0)" ) };
    const std::string roughness{ namedMaterialFloatExpression( model, material, index, "roughness", "0.0" ) };
    const std::string uroughness{ namedMaterialFloatExpression( model, material, index, "uroughness", "0.0" ) };
    const std::string vroughness{ namedMaterialFloatExpression( model, material, index, "vroughness", "0.0" ) };
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kr: " + kr );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kt: " + kt );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input roughness: " + roughness );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input uroughness: " + uroughness );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input vroughness: " + vroughness );
    return "::df::tint(\n"
           "                        "
           + kr
           + ",\n"
             "                        "
           + kt
           + ",\n"
             "                        ::df::microfacet_ggx_smith_bsdf(\n"
             "                            roughness_u: pbrt_mix_resolved_roughness("
           + roughness + ", " + uroughness
           + "),\n"
             "                            roughness_v: pbrt_mix_resolved_roughness("
           + roughness + ", " + vroughness
           + "),\n"
             "                            tint: color(1.0, 1.0, 1.0),\n"
             "                            mode: ::df::scatter_reflect_transmit))";
}

std::string namedMaterialMetalBsdfExpression( MdlMaterialModel&                   model,
                                              MdlTextureGraphGenerator&           textureGraph,
                                              const otk::pbrt::PbrtNamedMaterial& material,
                                              unsigned int                        index )
{
    const std::string eta{
        namedMaterialColorExpression( model, textureGraph, material, index, "eta", "color(0.2, 0.2, 0.2)" ) };
    const std::string k{
        namedMaterialColorExpression( model, textureGraph, material, index, "k", "color(3.0, 3.0, 3.0)" ) };
    const std::string roughness{ namedMaterialFloatExpression( model, material, index, "roughness", "0.1" ) };
    const std::string uroughness{ namedMaterialFloatExpression( model, material, index, "uroughness", "-1.0" ) };
    const std::string vroughness{ namedMaterialFloatExpression( model, material, index, "vroughness", "-1.0" ) };
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input eta: " + eta );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input k: " + k );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input roughness: " + roughness );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input uroughness: " + uroughness );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input vroughness: " + vroughness );
    return "::df::microfacet_ggx_smith_bsdf(\n"
           "                        roughness_u: pbrt_mix_resolved_roughness("
           + roughness + ", " + uroughness
           + "),\n"
             "                        roughness_v: pbrt_mix_resolved_roughness("
           + roughness + ", " + vroughness
           + "),\n"
             "                        tint: pbrt_mix_metal_conductor_tint("
           + eta + ", " + k
           + "),\n"
             "                        mode: ::df::scatter_reflect)";
}

std::string namedMaterialTranslucentBsdfExpression( MdlMaterialModel&                   model,
                                                    MdlTextureGraphGenerator&           textureGraph,
                                                    const otk::pbrt::PbrtNamedMaterial& material,
                                                    unsigned int                        index )
{
    const std::string kd{
        namedMaterialColorExpression( model, textureGraph, material, index, "Kd", "color(0.8, 0.8, 0.8)" ) };
    const std::string ks{
        namedMaterialColorExpression( model, textureGraph, material, index, "Ks", "color(0.0, 0.0, 0.0)" ) };
    const std::string reflect{
        namedMaterialColorExpression( model, textureGraph, material, index, "reflect", "color(0.5, 0.5, 0.5)" ) };
    const std::string transmit{
        namedMaterialColorExpression( model, textureGraph, material, index, "transmit", "color(0.5, 0.5, 0.5)" ) };
    const std::string roughness{ namedMaterialFloatExpression( model, material, index, "roughness", "0.1" ) };
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Kd: " + kd );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input Ks: " + ks );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input reflect: " + reflect );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input transmit: " + transmit );
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " input roughness: " + roughness );
    return "::df::color_normalized_mix(\n"
           "                        components: ::df::color_bsdf_component[](\n"
           "                            ::df::color_bsdf_component(\n"
           "                                weight: "
           + kd + " * " + reflect
           + ",\n"
             "                                component: ::df::diffuse_reflection_bsdf(\n"
             "                                    tint: color(1.0, 1.0, 1.0))),\n"
             "                            ::df::color_bsdf_component(\n"
             "                                weight: "
           + kd + " * " + transmit
           + ",\n"
             "                                component: ::df::diffuse_transmission_bsdf(\n"
             "                                    tint: color(1.0, 1.0, 1.0))),\n"
             "                            ::df::color_bsdf_component(\n"
             "                                weight: "
           + ks + " * " + reflect
           + ",\n"
             "                                component: ::df::simple_glossy_bsdf(\n"
             "                                    roughness_u: "
           + roughness
           + ",\n"
             "                                    roughness_v: "
           + roughness
           + ",\n"
             "                                    tint: color(1.0, 1.0, 1.0),\n"
             "                                    mode: ::df::scatter_reflect)),\n"
             "                            ::df::color_bsdf_component(\n"
             "                                weight: "
           + ks + " * " + transmit
           + ",\n"
             "                                component: ::df::simple_glossy_bsdf(\n"
             "                                    roughness_u: "
           + roughness
           + ",\n"
             "                                    roughness_v: "
           + roughness
           + ",\n"
             "                                    tint: color(1.0, 1.0, 1.0),\n"
             "                                    mode: ::df::scatter_transmit))))";
}

std::string unsupportedNamedMaterialBsdfExpression()
{
    return "::df::diffuse_reflection_bsdf(\n"
           "                        tint: color(1.0, 0.0, 1.0))";
}

std::string namedMaterialBsdfExpression( MdlMaterialModel&                   model,
                                         MdlTextureGraphGenerator&           textureGraph,
                                         GeneratedMdlSource&                 result,
                                         const otk::pbrt::PbrtNamedMaterial& material,
                                         unsigned int                        index )
{
    const std::string      type{ namedMaterialType( material ) };
    const PbrtMaterialKind kind{ pbrtMaterialKind( type ) };
    const std::string      typeComment{ type.empty() ? std::string{ "<empty>" } : type };
    model.comments.push_back( "pbrt named material " + std::to_string( index ) + " model: " + typeComment );

    switch( kind )
    {
        case PbrtMaterialKind::MATTE:
            return namedMaterialMatteBsdfExpression( model, textureGraph, material, index );
        case PbrtMaterialKind::PLASTIC:
            return namedMaterialPlasticBsdfExpression( model, textureGraph, material, index );
        case PbrtMaterialKind::SUBSTRATE:
            return namedMaterialSubstrateBsdfExpression( model, textureGraph, material, index );
        case PbrtMaterialKind::UBER:
            return namedMaterialUberBsdfExpression( model, textureGraph, material, index );
        case PbrtMaterialKind::MIRROR:
            return namedMaterialMirrorBsdfExpression( model, textureGraph, material, index );
        case PbrtMaterialKind::GLASS:
            return namedMaterialGlassBsdfExpression( model, textureGraph, material, index );
        case PbrtMaterialKind::METAL:
            return namedMaterialMetalBsdfExpression( model, textureGraph, material, index );
        case PbrtMaterialKind::TRANSLUCENT:
            return namedMaterialTranslucentBsdfExpression( model, textureGraph, material, index );
        default:
            appendUnsupportedReason( result, "Unsupported PBRT named material type " + typeComment );
            return unsupportedNamedMaterialBsdfExpression();
    }
}

MdlMaterialModel makeMatteMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "Kd", "color(0.8, 0.8, 0.8)" );
    appendMaterialParameter( model, "float", "sigma", "0.0" );
    appendMaterialParameter( model, "float", "alpha", "1.0" );
    appendMaterialParameter( model, "float", "opacity", "1.0" );

    const std::string kd{ textureGraph.materialColorExpression( material.params, "Kd", "color", "Kd" ) };
    const std::string alphaTexture{
        materialTextureCommentExpression( textureGraph, material.params, "alpha", "float" ) };
    const std::string shadowAlphaTexture{
        materialTextureCommentExpression( textureGraph, material.params, "shadowalpha", "float" ) };
    const std::string opacityTexture{
        materialTextureCommentExpression( textureGraph, material.params, "opacity", "float" ) };
    const std::string bumpmap{ materialBumpmapExpression( textureGraph, material.params ) };

    model.comments.push_back( "pbrt material model: matte" );
    model.comments.push_back( "pbrt material input Kd: " + kd );
    model.comments.push_back( "pbrt material input sigma: sigma" );
    model.comments.push_back( "pbrt material approximation: sigma degrees map to MDL Oren-Nayar roughness sigma / 90" );
    model.comments.push_back( "pbrt material input alpha: alpha; texture=" + alphaTexture );
    model.comments.push_back( "pbrt material input shadowalpha: any-hit texture=" + shadowAlphaTexture );
    model.comments.push_back( "pbrt material input opacity: opacity; texture=" + opacityTexture );
    appendBumpmapCommentsAndHelpers( model, bumpmap );
    model.helperDefinitions =
        "float pbrt_matte_sigma_roughness(float sigma_degrees) = ::math::clamp(sigma_degrees / 90.0, 0.0, 1.0);\n\n"
        + model.helperDefinitions;
    model.body =
        "    surface: material_surface(\n"
        "        scattering: ::df::diffuse_reflection_bsdf(\n"
        "            tint: "
        + kd + ",\n"
               "            roughness: pbrt_matte_sigma_roughness(sigma))),\n"
        + materialGeometryExpression( "alpha * opacity", bumpmap );
    return model;
}

MdlMaterialModel makePlasticMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "Kd", "color(0.8, 0.8, 0.8)" );
    appendMaterialParameter( model, "color", "Ks", "color(0.0, 0.0, 0.0)" );
    appendMaterialParameter( model, "float", "roughness", "0.1" );

    const std::string kd{ textureGraph.materialColorExpression( material.params, "Kd", "color", "Kd" ) };
    const std::string ks{ textureGraph.materialColorExpression( material.params, "Ks", "color", "Ks" ) };
    const std::string bumpmap{ materialBumpmapExpression( textureGraph, material.params ) };

    model.comments.push_back( "pbrt material model: plastic" );
    model.comments.push_back( "pbrt material input Kd: " + kd );
    model.comments.push_back( "pbrt material input Ks: " + ks );
    model.comments.push_back( "pbrt material input roughness: roughness" );
    appendBumpmapCommentsAndHelpers( model, bumpmap );
    appendRoughnessGapComment( model );
    model.comments.push_back(
        "pbrt material approximation: diffuse and glossy reflection use an MDL color-normalized mix" );
    model.body =
        "    surface: material_surface(\n"
        "        scattering: ::df::color_normalized_mix(\n"
        "            components: ::df::color_bsdf_component[](\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: " + kd + ",\n"
        "                    component: ::df::diffuse_reflection_bsdf(\n"
        "                        tint: color(1.0, 1.0, 1.0))),\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: " + ks + ",\n"
        "                    component: ::df::simple_glossy_bsdf(\n"
        "                        roughness_u: roughness,\n"
        "                        roughness_v: roughness,\n"
        "                        tint: color(1.0, 1.0, 1.0),\n"
        "                        mode: ::df::scatter_reflect)))))"
        + "\n";
    return model;
}

MdlMaterialModel makeUberMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "Kd", "color(0.8, 0.8, 0.8)" );
    appendMaterialParameter( model, "color", "Ks", "color(0.0, 0.0, 0.0)" );
    appendMaterialParameter( model, "color", "Kr", "color(0.0, 0.0, 0.0)" );
    appendMaterialParameter( model, "color", "Kt", "color(0.0, 0.0, 0.0)" );
    appendMaterialParameter( model, "float", "roughness", "0.1" );
    appendMaterialParameter( model, "float", "uroughness", "-1.0" );
    appendMaterialParameter( model, "float", "vroughness", "-1.0" );
    appendMaterialParameter( model, "float", "index", "1.5" );
    appendMaterialParameter( model, "float", "alpha", "1.0" );
    appendMaterialParameter( model, "color", "opacity", "color(1.0, 1.0, 1.0)" );

    const std::string kd{ textureGraph.materialColorExpression( material.params, "Kd", "color", "Kd" ) };
    const std::string ks{ textureGraph.materialColorExpression( material.params, "Ks", "color", "Ks" ) };
    const std::string kr{ textureGraph.materialColorExpression( material.params, "Kr", "color", "Kr" ) };
    const std::string kt{ textureGraph.materialColorExpression( material.params, "Kt", "color", "Kt" ) };
    const std::string alphaTexture{
        materialTextureCommentExpression( textureGraph, material.params, "alpha", "float" ) };
    const std::string opacityTexture{
        materialTextureCommentExpression( textureGraph, material.params, "opacity", "float" ) };
    const std::string bumpmap{ materialBumpmapExpression( textureGraph, material.params ) };

    model.comments.push_back( "pbrt material model: uber" );
    model.comments.push_back( "pbrt material input Kd: " + kd );
    model.comments.push_back( "pbrt material input Ks: " + ks );
    model.comments.push_back( "pbrt material input Kr: " + kr );
    model.comments.push_back( "pbrt material input Kt: " + kt );
    model.comments.push_back( "pbrt material input roughness: roughness" );
    model.comments.push_back( "pbrt material input uroughness: uroughness" );
    model.comments.push_back( "pbrt material input vroughness: vroughness" );
    model.comments.push_back( "pbrt material input index: index" );
    model.comments.push_back( "pbrt material input alpha: alpha; texture=" + alphaTexture );
    model.comments.push_back( "pbrt material input opacity: opacity; texture=" + opacityTexture );
    appendBumpmapCommentsAndHelpers( model, bumpmap );
    appendRoughnessGapComment( model );
    model.comments.push_back( "pbrt material approximation: PBRT uber lobes use an MDL color-normalized mix" );
    model.comments.push_back(
        "pbrt material approximation: spectrum opacity weights BSDF lobes and adds transparent transmission; alpha "
        "remains "
        "cutout" );
    model.helperDefinitions =
        "float pbrt_uber_resolved_roughness(float roughness, float axis_roughness) = "
        "axis_roughness >= 0.0 ? axis_roughness : roughness;\n\n"
        "color pbrt_uber_clamped_opacity(color opacity) = "
        "::math::clamp(opacity, color(0.0, 0.0, 0.0), color(1.0, 1.0, 1.0));\n\n"
        "color pbrt_uber_opacity_weight(color opacity) = pbrt_uber_clamped_opacity(opacity);\n\n"
        "color pbrt_uber_transparency_weight(color opacity) = "
        "color(1.0, 1.0, 1.0) - pbrt_uber_clamped_opacity(opacity);\n\n"
        + model.helperDefinitions;
    model.body = std::string{ "    ior: color(index, index, index),\n"
                              "    surface: material_surface(\n"
                              "        scattering: ::df::color_normalized_mix(\n"
                              "            components: ::df::color_bsdf_component[](\n"
                              "                ::df::color_bsdf_component(\n"
                              "                    weight: " }
                 + "pbrt_uber_opacity_weight(opacity) * " + kd
                 + ",\n"
                   "                    component: ::df::diffuse_reflection_bsdf(\n"
                   "                        tint: color(1.0, 1.0, 1.0))),\n"
                   "                ::df::color_bsdf_component(\n"
                   "                    weight: "
                 + "pbrt_uber_opacity_weight(opacity) * " + ks
                 + ",\n"
                   "                    component: ::df::simple_glossy_bsdf(\n"
                   "                        roughness_u: pbrt_uber_resolved_roughness(roughness, uroughness),\n"
                   "                        roughness_v: pbrt_uber_resolved_roughness(roughness, vroughness),\n"
                   "                        tint: color(1.0, 1.0, 1.0),\n"
                   "                        mode: ::df::scatter_reflect)),\n"
                   "                ::df::color_bsdf_component(\n"
                   "                    weight: pbrt_uber_opacity_weight(opacity) * "
                 + kr
                 + ",\n"
                   "                    component: ::df::specular_bsdf(\n"
                   "                        tint: color(1.0, 1.0, 1.0),\n"
                   "                        mode: ::df::scatter_reflect)),\n"
                   "                ::df::color_bsdf_component(\n"
                   "                    weight: pbrt_uber_opacity_weight(opacity) * "
                 + kt
                 + ",\n"
                   "                    component: ::df::specular_bsdf(\n"
                   "                        tint: color(1.0, 1.0, 1.0),\n"
                   "                        mode: ::df::scatter_transmit)),\n"
                   "                ::df::color_bsdf_component(\n"
                   "                    weight: pbrt_uber_transparency_weight(opacity),\n"
                   "                    component: ::df::specular_bsdf(\n"
                   "                        tint: color(1.0, 1.0, 1.0),\n"
                   "                        mode: ::df::scatter_transmit))))),\n"
        + materialGeometryExpression( "alpha", bumpmap );
    return model;
}

MdlMaterialModel makeMirrorMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "Kr", "color(1.0, 1.0, 1.0)" );

    const std::string kr{ textureGraph.materialColorExpression( material.params, "Kr", "color", "Kr" ) };

    model.comments.push_back( "pbrt material model: mirror" );
    model.comments.push_back( "pbrt material input Kr: " + kr );
    model.body =
        "    surface: material_surface(\n"
        "        scattering: ::df::specular_bsdf(\n"
        "            tint: " + kr + ",\n"
        "            mode: ::df::scatter_reflect))\n";
    return model;
}

MdlMaterialModel makeGlassMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "Kr", "color(1.0, 1.0, 1.0)" );
    appendMaterialParameter( model, "color", "Kt", "color(1.0, 1.0, 1.0)" );
    appendMaterialParameter( model, "float", "index", "1.5" );
    appendMaterialParameter( model, "float", "roughness", "0.0" );
    appendMaterialParameter( model, "float", "uroughness", "0.0" );
    appendMaterialParameter( model, "float", "vroughness", "0.0" );

    const std::string kr{ textureGraph.materialColorExpression( material.params, "Kr", "color", "Kr" ) };
    const std::string kt{ textureGraph.materialColorExpression( material.params, "Kt", "color", "Kt" ) };

    model.comments.push_back( "pbrt material model: glass" );
    model.comments.push_back( "pbrt material input Kr: " + kr );
    model.comments.push_back( "pbrt material input Kt: " + kt );
    model.comments.push_back( "pbrt material input index/eta: index" );
    model.comments.push_back( "pbrt material input roughness: roughness" );
    model.comments.push_back( "pbrt material input uroughness: uroughness" );
    model.comments.push_back( "pbrt material input vroughness: vroughness" );
    model.comments.push_back( "pbrt material approximation: rough glass uses an MDL GGX microfacet dielectric lobe" );
    appendRoughnessGapComment( model );
    model.helperDefinitions =
        "float pbrt_glass_resolved_roughness(float roughness, float axis_roughness) = "
        "axis_roughness > 0.0 ? axis_roughness : roughness;\n\n";
    model.body =
        "    ior: color(index, index, index),\n"
        "    surface: material_surface(\n"
        "        scattering: ::df::tint(\n"
        "            "
        + kr + ",\n"
        "            " + kt
        + ",\n"
          "            ::df::microfacet_ggx_smith_bsdf(\n"
          "                roughness_u: pbrt_glass_resolved_roughness(roughness, uroughness),\n"
          "                roughness_v: pbrt_glass_resolved_roughness(roughness, vroughness),\n"
          "                tint: color(1.0, 1.0, 1.0),\n"
          "                mode: ::df::scatter_reflect_transmit)))\n";
    return model;
}

MdlMaterialModel makeMetalMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "eta", "color(0.2, 0.2, 0.2)" );
    appendMaterialParameter( model, "color", "k", "color(3.0, 3.0, 3.0)" );
    appendMaterialParameter( model, "float", "roughness", "0.1" );
    appendMaterialParameter( model, "float", "uroughness", "-1.0" );
    appendMaterialParameter( model, "float", "vroughness", "-1.0" );

    const std::string eta{ textureGraph.materialColorExpression( material.params, "eta", "color", "eta" ) };
    const std::string k{ textureGraph.materialColorExpression( material.params, "k", "color", "k" ) };

    model.comments.push_back( "pbrt material model: metal" );
    model.comments.push_back( "pbrt material input eta: " + eta );
    model.comments.push_back( "pbrt material input k: " + k );
    model.comments.push_back( "pbrt material input roughness: roughness" );
    model.comments.push_back( "pbrt material input uroughness: uroughness" );
    model.comments.push_back( "pbrt material input vroughness: vroughness" );
    model.comments.push_back( "pbrt material gap: PBRT-exact spectral conductor behavior is approximated" );
    appendRoughnessGapComment( model );
    model.comments.push_back(
        "pbrt material approximation: RGB eta/k maps to MDL microfacet tint using normal-incidence conductor "
        "reflectance" );
    model.helperDefinitions =
        "float pbrt_metal_resolved_roughness(float roughness, float axis_roughness) = "
        "axis_roughness >= 0.0 ? axis_roughness : roughness;\n\n"
        "color pbrt_metal_conductor_tint(color eta, color k) =\n"
        "    ((eta - color(1.0, 1.0, 1.0)) * (eta - color(1.0, 1.0, 1.0)) + k * k) /\n"
        "    ((eta + color(1.0, 1.0, 1.0)) * (eta + color(1.0, 1.0, 1.0)) + k * k);\n\n";
    model.body =
        "    surface: material_surface(\n"
        "        scattering: ::df::microfacet_ggx_smith_bsdf(\n"
        "            roughness_u: pbrt_metal_resolved_roughness(roughness, uroughness),\n"
        "            roughness_v: pbrt_metal_resolved_roughness(roughness, vroughness),\n"
        "            tint: pbrt_metal_conductor_tint("
        + eta + ", " + k
        + "),\n"
          "            mode: ::df::scatter_reflect))\n";
    return model;
}

MdlMaterialModel makeSubstrateMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "Kd", "color(0.5, 0.5, 0.5)" );
    appendMaterialParameter( model, "color", "Ks", "color(0.5, 0.5, 0.5)" );
    appendMaterialParameter( model, "float", "roughness", "0.1" );
    appendMaterialParameter( model, "float", "uroughness", "-1.0" );
    appendMaterialParameter( model, "float", "vroughness", "-1.0" );

    const std::string kd{ textureGraph.materialColorExpression( material.params, "Kd", "color", "Kd" ) };
    const std::string ks{ textureGraph.materialColorExpression( material.params, "Ks", "color", "Ks" ) };
    const std::string bumpmap{ materialBumpmapExpression( textureGraph, material.params ) };

    model.comments.push_back( "pbrt material model: substrate" );
    model.comments.push_back( "pbrt material input Kd: " + kd );
    model.comments.push_back( "pbrt material input Ks: " + ks );
    model.comments.push_back( "pbrt material input roughness: roughness" );
    model.comments.push_back( "pbrt material input uroughness: uroughness" );
    model.comments.push_back( "pbrt material input vroughness: vroughness" );
    appendBumpmapCommentsAndHelpers( model, bumpmap );
    appendRoughnessGapComment( model );
    model.comments.push_back(
        "pbrt material approximation: diffuse base and glossy layer use an MDL color-weighted layer" );
    model.helperDefinitions =
        "float pbrt_substrate_resolved_roughness(float roughness, float axis_roughness) = "
        "axis_roughness >= 0.0 ? axis_roughness : roughness;\n\n"
        + model.helperDefinitions;
    model.body =
        "    surface: material_surface(\n"
        "        scattering: ::df::color_weighted_layer(\n"
        "            weight: " + ks + ",\n"
        "            layer: ::df::simple_glossy_bsdf(\n"
        "                roughness_u: pbrt_substrate_resolved_roughness(roughness, uroughness),\n"
        "                roughness_v: pbrt_substrate_resolved_roughness(roughness, vroughness),\n"
        "                tint: color(1.0, 1.0, 1.0),\n"
        "                mode: ::df::scatter_reflect),\n"
        "            base: ::df::diffuse_reflection_bsdf(\n"
        "                tint: "
        + kd + ")))"
        + "\n";
    return model;
}

MdlMaterialModel makeTranslucentMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "Kd", "color(0.8, 0.8, 0.8)" );
    appendMaterialParameter( model, "color", "Ks", "color(0.0, 0.0, 0.0)" );
    appendMaterialParameter( model, "color", "reflect", "color(0.5, 0.5, 0.5)" );
    appendMaterialParameter( model, "color", "transmit", "color(0.5, 0.5, 0.5)" );
    appendMaterialParameter( model, "float", "roughness", "0.1" );
    appendMaterialParameter( model, "color", "opacity", "color(1.0, 1.0, 1.0)" );

    const std::string kd{ textureGraph.materialColorExpression( material.params, "Kd", "color", "Kd" ) };
    const std::string ks{ textureGraph.materialColorExpression( material.params, "Ks", "color", "Ks" ) };
    const std::string reflect{ textureGraph.materialColorExpression( material.params, "reflect", "color", "reflect" ) };
    const std::string transmit{
        textureGraph.materialColorExpression( material.params, "transmit", "color", "transmit" ) };
    const std::string opacityTexture{
        materialTextureCommentExpression( textureGraph, material.params, "opacity", "float" ) };
    const std::string bumpmap{ materialBumpmapExpression( textureGraph, material.params ) };

    model.comments.push_back( "pbrt material model: translucent" );
    model.comments.push_back( "pbrt material input Kd: " + kd );
    model.comments.push_back( "pbrt material input Ks: " + ks );
    model.comments.push_back( "pbrt material input reflect: " + reflect );
    model.comments.push_back( "pbrt material input transmit: " + transmit );
    model.comments.push_back( "pbrt material input roughness: roughness" );
    model.comments.push_back( "pbrt material input opacity: opacity; texture=" + opacityTexture );
    appendBumpmapCommentsAndHelpers( model, bumpmap );
    model.comments.push_back( "pbrt material input eta: fixed 1.5" );
    appendRoughnessGapComment( model );
    model.comments.push_back(
        "pbrt material approximation: diffuse/glossy reflection and transmission use an MDL color-normalized mix" );
    model.comments.push_back(
        "pbrt material approximation: spectrum opacity weights generated translucent lobes and adds transparent "
        "transmission" );
    model.helperDefinitions =
        "color pbrt_translucent_clamped_opacity(color opacity) = "
        "::math::clamp(opacity, color(0.0, 0.0, 0.0), color(1.0, 1.0, 1.0));\n\n"
        "color pbrt_translucent_opacity_weight(color opacity) = pbrt_translucent_clamped_opacity(opacity);\n\n"
        "color pbrt_translucent_transparency_weight(color opacity) = "
        "color(1.0, 1.0, 1.0) - pbrt_translucent_clamped_opacity(opacity);\n\n"
        + model.helperDefinitions;
    model.body =
        "    ior: color(1.5, 1.5, 1.5),\n"
        "    surface: material_surface(\n"
        "        scattering: ::df::color_normalized_mix(\n"
        "            components: ::df::color_bsdf_component[](\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: pbrt_translucent_opacity_weight(opacity) * " + kd + " * " + reflect + ",\n"
        "                    component: ::df::diffuse_reflection_bsdf(\n"
        "                        tint: color(1.0, 1.0, 1.0))),\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: pbrt_translucent_opacity_weight(opacity) * " + kd + " * " + transmit + ",\n"
        "                    component: ::df::diffuse_transmission_bsdf(\n"
        "                        tint: color(1.0, 1.0, 1.0))),\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: pbrt_translucent_opacity_weight(opacity) * " + ks + " * " + reflect + ",\n"
        "                    component: ::df::simple_glossy_bsdf(\n"
        "                        roughness_u: roughness,\n"
        "                        roughness_v: roughness,\n"
        "                        tint: color(1.0, 1.0, 1.0),\n"
        "                        mode: ::df::scatter_reflect)),\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: pbrt_translucent_opacity_weight(opacity) * " + ks + " * " + transmit + ",\n"
        "                    component: ::df::simple_glossy_bsdf(\n"
        "                        roughness_u: roughness,\n"
        "                        roughness_v: roughness,\n"
        "                        tint: color(1.0, 1.0, 1.0),\n"
        "                        mode: ::df::scatter_transmit)),\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: pbrt_translucent_transparency_weight(opacity),\n"
        "                    component: ::df::specular_bsdf(\n"
        "                        tint: color(1.0, 1.0, 1.0),\n"
        "                        mode: ::df::scatter_transmit)))))"
        + ( hasBumpmapExpression( bumpmap ) ? std::string{ ",\n" } + materialGeometryExpression( "", bumpmap ) : "\n" );
    return model;
}

MdlMaterialModel makeSubsurfaceMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "Kr", "color(1.0, 1.0, 1.0)" );
    appendMaterialParameter( model, "color", "Kt", "color(1.0, 1.0, 1.0)" );
    appendMaterialParameter( model, "color", "sigma_a", "color(0.0011, 0.0024, 0.014)" );
    appendMaterialParameter( model, "color", "sigma_s", "color(2.55, 3.21, 3.77)" );
    appendMaterialParameter( model, "float", "scale", "1.0" );
    appendMaterialParameter( model, "float", "g", "0.0" );
    appendMaterialParameter( model, "float", "eta", "1.33" );
    appendMaterialParameter( model, "float", "uroughness", "0.0" );
    appendMaterialParameter( model, "float", "vroughness", "0.0" );

    const std::string kr{ textureGraph.materialColorExpression( material.params, "Kr", "color", "Kr" ) };
    const std::string kt{ textureGraph.materialColorExpression( material.params, "Kt", "color", "Kt" ) };
    const std::string sigmaA{ textureGraph.materialColorExpression( material.params, "sigma_a", "color", "sigma_a" ) };
    const std::string sigmaS{ textureGraph.materialColorExpression( material.params, "sigma_s", "color", "sigma_s" ) };
    const std::string bumpmap{ materialBumpmapExpression( textureGraph, material.params ) };
    const std::string albedo{ "pbrt_subsurface_albedo(" + sigmaA + ", " + sigmaS + ", scale)" };

    model.comments.push_back( "pbrt material model: subsurface" );
    model.comments.push_back( "pbrt material input Kr: " + kr );
    model.comments.push_back( "pbrt material input Kt: " + kt );
    model.comments.push_back( "pbrt material input sigma_a: " + sigmaA );
    model.comments.push_back( "pbrt material input sigma_s: " + sigmaS );
    model.comments.push_back( "pbrt material input scale: scale" );
    model.comments.push_back( "pbrt material input g: g" );
    model.comments.push_back( "pbrt material input eta: eta" );
    model.comments.push_back( "pbrt material input uroughness: uroughness" );
    model.comments.push_back( "pbrt material input vroughness: vroughness" );
    model.comments.push_back( "pbrt material input name: named scattering database lookup is not modeled" );
    appendBumpmapCommentsAndHelpers( model, bumpmap );
    model.comments.push_back(
        "pbrt material gap: full PBRT BSSRDF transport and named-medium scattering data are not evaluated" );
    model.comments.push_back(
        "pbrt material approximation: sigma_a/sigma_s albedo drives diffuse reflection and transmission lobes" );
    model.helperDefinitions =
        "color pbrt_subsurface_albedo(color sigma_a, color sigma_s, float scale) =\n"
        "    ::math::clamp((sigma_s * scale) / ((sigma_a + sigma_s) * scale + color(0.000001, 0.000001, 0.000001)), "
        "color(0.0, 0.0, 0.0), color(1.0, 1.0, 1.0));\n\n"
        + model.helperDefinitions;
    model.body =
        "    ior: color(eta, eta, eta),\n"
        "    surface: material_surface(\n"
        "        scattering: ::df::color_normalized_mix(\n"
        "            components: ::df::color_bsdf_component[](\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: " + kr + " * " + albedo + ",\n"
        "                    component: ::df::diffuse_reflection_bsdf(\n"
        "                        tint: color(1.0, 1.0, 1.0))),\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: " + kt + " * " + albedo + ",\n"
        "                    component: ::df::diffuse_transmission_bsdf(\n"
        "                        tint: color(1.0, 1.0, 1.0))))))"
        + ( hasBumpmapExpression( bumpmap ) ? std::string{ ",\n" } + materialGeometryExpression( "", bumpmap ) : "\n" );
    return model;
}

MdlMaterialModel makeKdSubsurfaceMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "Kd", "color(0.5, 0.5, 0.5)" );
    appendMaterialParameter( model, "color", "Kr", "color(1.0, 1.0, 1.0)" );
    appendMaterialParameter( model, "color", "Kt", "color(1.0, 1.0, 1.0)" );
    appendMaterialParameter( model, "color", "mfp", "color(1.0, 1.0, 1.0)" );
    appendMaterialParameter( model, "float", "scale", "1.0" );
    appendMaterialParameter( model, "float", "g", "0.0" );
    appendMaterialParameter( model, "float", "eta", "1.33" );
    appendMaterialParameter( model, "float", "uroughness", "0.0" );
    appendMaterialParameter( model, "float", "vroughness", "0.0" );

    const std::string kd{ textureGraph.materialColorExpression( material.params, "Kd", "color", "Kd" ) };
    const std::string kr{ textureGraph.materialColorExpression( material.params, "Kr", "color", "Kr" ) };
    const std::string kt{ textureGraph.materialColorExpression( material.params, "Kt", "color", "Kt" ) };
    const std::string mfp{ textureGraph.materialColorExpression( material.params, "mfp", "color", "mfp" ) };
    const std::string bumpmap{ materialBumpmapExpression( textureGraph, material.params ) };

    model.comments.push_back( "pbrt material model: kdsubsurface" );
    model.comments.push_back( "pbrt material input Kd: " + kd );
    model.comments.push_back( "pbrt material input Kr: " + kr );
    model.comments.push_back( "pbrt material input Kt: " + kt );
    model.comments.push_back( "pbrt material input mfp: " + mfp );
    model.comments.push_back( "pbrt material input scale: scale" );
    model.comments.push_back( "pbrt material input g: g" );
    model.comments.push_back( "pbrt material input eta: eta" );
    model.comments.push_back( "pbrt material input uroughness: uroughness" );
    model.comments.push_back( "pbrt material input vroughness: vroughness" );
    appendBumpmapCommentsAndHelpers( model, bumpmap );
    model.comments.push_back( "pbrt material gap: full PBRT diffusion-profile BSSRDF transport is not evaluated" );
    model.comments.push_back( "pbrt material approximation: Kd drives diffuse reflection and transmission lobes" );
    model.body =
        "    ior: color(eta, eta, eta),\n"
        "    surface: material_surface(\n"
        "        scattering: ::df::color_normalized_mix(\n"
        "            components: ::df::color_bsdf_component[](\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: " + kr + " * " + kd + ",\n"
        "                    component: ::df::diffuse_reflection_bsdf(\n"
        "                        tint: color(1.0, 1.0, 1.0))),\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: " + kt + " * " + kd + ",\n"
        "                    component: ::df::diffuse_transmission_bsdf(\n"
        "                        tint: color(1.0, 1.0, 1.0))))))"
        + ( hasBumpmapExpression( bumpmap ) ? std::string{ ",\n" } + materialGeometryExpression( "", bumpmap ) : "\n" );
    return model;
}

std::string mixNamedMaterialBsdfExpression( MdlMaterialModel&              model,
                                            MdlTextureGraphGenerator&      textureGraph,
                                            GeneratedMdlSource&            result,
                                            const otk::pbrt::PbrtMaterial& material,
                                            const std::string&             paramName,
                                            unsigned int                   index )
{
    const std::string materialName{ material.params.FindOneString( paramName, std::string{} ) };
    if( materialName.empty() )
    {
        model.comments.push_back( "pbrt material input " + paramName + ": missing" );
        appendUnsupportedReason( result, "Missing PBRT mix " + paramName );
        return unsupportedNamedMaterialBsdfExpression();
    }

    model.comments.push_back( "pbrt material input " + paramName + ": named material " + std::to_string( index ) );
    const otk::pbrt::PbrtNamedMaterialMap::const_iterator namedMaterial = material.graph.namedMaterials.find( materialName );
    if( namedMaterial == material.graph.namedMaterials.end() )
    {
        appendUnsupportedReason( result, "Missing PBRT named material reference" );
        return unsupportedNamedMaterialBsdfExpression();
    }

    return namedMaterialBsdfExpression( model, textureGraph, result, namedMaterial->second, index );
}

MdlMaterialModel makeMixMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph, GeneratedMdlSource& result )
{
    MdlMaterialModel model;
    appendMaterialParameter( model, "color", "amount", "color(0.5, 0.5, 0.5)" );

    model.comments.push_back( "pbrt material model: mix" );
    const std::string first{ mixNamedMaterialBsdfExpression( model, textureGraph, result, material, "namedmaterial1", 0U ) };
    const std::string second{ mixNamedMaterialBsdfExpression( model, textureGraph, result, material, "namedmaterial2", 1U ) };
    const std::string amountTexture{
        materialTextureCommentExpression( textureGraph, material.params, "amount", "color" ) };

    model.comments.push_back( "pbrt material input amount: amount; texture=" + amountTexture );
    model.comments.push_back( "pbrt material weighting: namedmaterial1 uses amount; namedmaterial2 uses 1 - amount" );
    model.comments.push_back( "pbrt material approximation: mix composes supported named material MDL closures" );
    model.helperDefinitions =
        "float pbrt_mix_matte_sigma_roughness(float sigma_degrees) = ::math::clamp(sigma_degrees / 90.0, 0.0, 1.0);\n\n"
        "float pbrt_mix_resolved_roughness(float roughness, float axis_roughness) = "
        "axis_roughness >= 0.0 ? axis_roughness : roughness;\n\n"
        "float pbrt_mix_clamped_opacity(float opacity) = ::math::clamp(opacity, 0.0, 1.0);\n\n"
        "color pbrt_mix_opacity_weight(float opacity) =\n"
        "    color(pbrt_mix_clamped_opacity(opacity), pbrt_mix_clamped_opacity(opacity), "
        "pbrt_mix_clamped_opacity(opacity));\n\n"
        "color pbrt_mix_transparency_weight(float opacity) =\n"
        "    color(1.0 - pbrt_mix_clamped_opacity(opacity), 1.0 - pbrt_mix_clamped_opacity(opacity), "
        "1.0 - pbrt_mix_clamped_opacity(opacity));\n\n"
        "color pbrt_mix_metal_conductor_tint(color eta, color k) =\n"
        "    ((eta - color(1.0, 1.0, 1.0)) * (eta - color(1.0, 1.0, 1.0)) + k * k) /\n"
        "    ((eta + color(1.0, 1.0, 1.0)) * (eta + color(1.0, 1.0, 1.0)) + k * k);\n\n";
    model.body =
        "    surface: material_surface(\n"
        "        scattering: ::df::color_normalized_mix(\n"
        "            components: ::df::color_bsdf_component[](\n"
        "                ::df::color_bsdf_component(\n"
        "                    weight: amount,\n"
        "                    component: "
        + first
        + "),\n"
          "                ::df::color_bsdf_component(\n"
          "                    weight: color(1.0, 1.0, 1.0) - amount,\n"
          "                    component: "
        + second + "))))\n";
    return model;
}

MdlMaterialModel makeUnsupportedMaterialModel( const otk::pbrt::PbrtMaterial& material,
                                               PbrtMaterialKind                kind,
                                               GeneratedMdlSource&            result )
{
    const std::string type{ material.type.empty() ? std::string{ "<empty>" } : material.type };

    MdlMaterialModel model;
    model.comments.push_back( "pbrt material model: " + type );
    const PbrtMaterialGapPolicy* const policy{ explicitMaterialGapPolicy( kind ) };
    if( policy != nullptr )
    {
        model.comments.push_back( "pbrt material gap policy: " + policy->policy );
        model.comments.push_back( "pbrt material gap coverage: " + policy->coverageReason );
        appendUnsupportedReason( result, "Explicit PBRT material gap " + type + ": " + policy->policy );
        if( kind == PbrtMaterialKind::FOURIER )
        {
            if( hasFourierBsdfFile( material ) )
            {
                model.comments.push_back( "pbrt fourier bsdffile: preserved as material metadata" );
            }
            else
            {
                model.comments.push_back( "pbrt fourier bsdffile: missing" );
                appendUnsupportedReason( result, "PBRT Fourier material missing bsdffile" );
            }
        }
    }
    else
    {
        model.comments.push_back( "pbrt material gap policy: unknown material type" );
        appendUnsupportedReason( result, "Unsupported PBRT material type " + type );
    }
    model.body =
        "    surface: material_surface(\n"
        "        scattering: ::df::diffuse_reflection_bsdf(\n"
        "            tint: color(1.0, 0.0, 1.0)))\n";
    return model;
}

MdlMaterialModel makeMaterialModel( const otk::pbrt::PbrtMaterial& material, MdlTextureGraphGenerator& textureGraph, GeneratedMdlSource& result )
{
    const PbrtMaterialKind kind{ pbrtMaterialKind( material.type ) };
    switch( kind )
    {
        case PbrtMaterialKind::MATTE:
            return makeMatteMaterialModel( material, textureGraph );
        case PbrtMaterialKind::PLASTIC:
            return makePlasticMaterialModel( material, textureGraph );
        case PbrtMaterialKind::UBER:
            return makeUberMaterialModel( material, textureGraph );
        case PbrtMaterialKind::MIRROR:
            return makeMirrorMaterialModel( material, textureGraph );
        case PbrtMaterialKind::GLASS:
            return makeGlassMaterialModel( material, textureGraph );
        case PbrtMaterialKind::METAL:
            return makeMetalMaterialModel( material, textureGraph );
        case PbrtMaterialKind::SUBSTRATE:
            return makeSubstrateMaterialModel( material, textureGraph );
        case PbrtMaterialKind::TRANSLUCENT:
            return makeTranslucentMaterialModel( material, textureGraph );
        case PbrtMaterialKind::SUBSURFACE:
            return makeSubsurfaceMaterialModel( material, textureGraph );
        case PbrtMaterialKind::KD_SUBSURFACE:
            return makeKdSubsurfaceMaterialModel( material, textureGraph );
        case PbrtMaterialKind::MIX:
            return makeMixMaterialModel( material, textureGraph, result );
        default:
            return makeUnsupportedMaterialModel( material, kind, result );
    }
}

}  // namespace

GeneratedMdlSource generateMdlSource( const MdlShaderKey& key )
{
    const std::string suffix{ stableHash( key.signature ) };

    GeneratedMdlSource result;
    result.moduleName   = "::otk::demand_pbrt_scene::pbrt_" + suffix;
    result.materialName = "material_" + suffix;

    std::ostringstream source;
    source << "mdl 1.10;\n"
           << "import ::df::*;\n"
           << "import ::math::*;\n"
           << "\n"
           << "export material " << result.materialName << "() = material(\n"
           << "    surface: material_surface(\n"
           << "        scattering: ::df::diffuse_reflection_bsdf(\n"
           << "            tint: color(0.8, 0.8, 0.8))));\n";
    result.source = source.str();
    return result;
}

GeneratedMdlSource generateMdlSource( const otk::pbrt::PbrtMaterial& material )
{
    const MdlShaderKey key{ makeMdlShaderKey( material ) };
    const std::string  suffix{ stableHash( key.signature ) };

    GeneratedMdlSource result;
    result.moduleName   = "::otk::demand_pbrt_scene::pbrt_" + suffix;
    result.materialName = "material_" + suffix;

    MdlTextureGraphGenerator textureGraph{ material.graph, result };
    const MdlMaterialModel   materialModel{ makeMaterialModel( material, textureGraph, result ) };

    std::ostringstream source;
    source << "mdl 1.10;\n"
           << "import ::df::*;\n"
           << "import ::math::*;\n"
           << "import ::state::*;\n"
           << "\n";
    for( std::vector<std::string>::const_iterator it = materialModel.comments.begin(); it != materialModel.comments.end(); ++it )
    {
        source << "// " << *it << "\n";
    }
    if( !materialModel.comments.empty() )
    {
        source << "\n";
    }
    for( std::vector<std::string>::const_iterator it = result.unsupportedReasons.begin();
         it != result.unsupportedReasons.end(); ++it )
    {
        source << "// unsupported: " << *it << "\n";
    }
    if( !result.unsupportedReasons.empty() )
    {
        source << "\n";
    }
    source << textureGraph.sourcePreamble() << materialModel.helperDefinitions << textureGraph.functionDefinitions()
           << "export material " << result.materialName << mdlParameterList( materialModel.parameters ) << " = material(\n"
           << materialModel.body << ");\n";
    result.source = source.str();
    return result;
}

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL

