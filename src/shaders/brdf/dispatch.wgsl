// src/shaders/brdf/dispatch.wgsl
// BRDF dispatch table to route between shading models at runtime
// Exists to centralize BRDF selection and parameter preparation
// RELEVANT FILES: src/shaders/brdf/common.wgsl, src/shaders/brdf/*, src/shaders/lighting.wgsl, src/lighting/types.rs

#include "brdf/common.wgsl"
#include "brdf/lambert.wgsl"
#include "brdf/phong.wgsl"
#include "brdf/oren_nayar.wgsl"
#include "brdf/cook_torrance.wgsl"
#include "brdf/disney_principled.wgsl"
#include "brdf/ashikhmin_shirley.wgsl"
#include "brdf/ward.wgsl"
#include "brdf/toon.wgsl"
#include "brdf/minnaert.wgsl"

fn eval_brdf(
    normal: vec3<f32>,
    view: vec3<f32>,
    light: vec3<f32>,
    base_color: vec3<f32>,
    params: ShadingParamsGPU,
) -> vec3<f32> {
    switch (params.brdf) {
        case BRDF_LAMBERT: {
            return brdf_lambert(base_color);
        }
        case BRDF_PHONG: {
            return brdf_phong(normal, view, light, base_color, params);
        }
        case BRDF_BLINN_PHONG: {
            return brdf_phong(normal, view, light, base_color, params);
        }
        case BRDF_OREN_NAYAR: {
            return brdf_oren_nayar(normal, view, light, base_color, params);
        }
        case BRDF_COOK_TORRANCE_GGX: {
            return brdf_cook_torrance_ggx(normal, view, light, base_color, params);
        }
        case BRDF_COOK_TORRANCE_BECKMANN: {
            return brdf_cook_torrance_beckmann(normal, view, light, base_color, params);
        }
        case BRDF_DISNEY_PRINCIPLED: {
            return brdf_disney_principled(normal, view, light, base_color, params);
        }
        case BRDF_ASHIKHMIN_SHIRLEY: {
            return brdf_ashikhmin_shirley(normal, view, light, base_color, params);
        }
        case BRDF_WARD: {
            return brdf_ward(normal, view, light, base_color, params);
        }
        case BRDF_TOON: {
            return brdf_toon(normal, view, light, base_color, params);
        }
        case BRDF_MINNAERT: {
            return brdf_minnaert(normal, view, light, base_color, params);
        }
        case BRDF_SUBSURFACE: {
            return brdf_disney_principled(normal, view, light, base_color, params);
        }
        case BRDF_HAIR: {
            return brdf_ashikhmin_shirley(normal, view, light, base_color, params);
        }
        default: {
            return brdf_lambert(base_color);
        }
    }
}
