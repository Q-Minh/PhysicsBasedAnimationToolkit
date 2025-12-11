#include "Params.h"

#include <nanobind/nanobind.h>
#include <pbat/io/Archive.h>
#include <pbat/sim/contact/ogc/Enums.h>
#include <pbat/sim/contact/ogc/Params.h>

namespace pbat::py::sim::contact::ogc {

void BindParams(nanobind::module_& m)
{
    namespace nb                = nanobind;
    using ScalarType            = Scalar;
    using ParamsType            = pbat::sim::contact::ogc::Params<ScalarType>;
    using EBuildQuality         = pbat::sim::contact::ogc::EBuildQuality;
    using ESceneFeatures        = pbat::sim::contact::ogc::ESceneFeatures;

    nb::enum_<EBuildQuality>(m, "BuildQuality")
        .value("Low", EBuildQuality::Low, "Low build quality (fast build time)")
        .value("Medium", EBuildQuality::Medium, "Medium build quality (balanced)")
        .value("High", EBuildQuality::High, "High build quality (slow build time)")
        .value("Refit", EBuildQuality::Refit, "Refit")
        .export_values();

    nb::enum_<ESceneFeatures>(m, "SceneFeatures")
        .value("None", ESceneFeatures::None, "No special features")
        .value("Dynamic", ESceneFeatures::Dynamic, "Dynamic scene")
        .value("Compact", ESceneFeatures::Compact, "Compact representation")
        .value("Robust", ESceneFeatures::Robust, "Robust representation")
        .export_values();

    nb::class_<ParamsType>(m, "Params")
        .def(
            nb::init<>(),
            "Construct default OGC parameters.")
        .def_rw(
            "r",
            &ParamsType::r,
            "(float) Contact radius.")
        .def_rw(
            "rq",
            &ParamsType::rq,
            "(float) Query inflation radius.")
        .def_rw(
            "scene_features",
            &ParamsType::eSceneFeatures,
            "(SceneFeatures) Scene features.")
        .def_rw(
            "dynamic_scene_bvh_quality",
            &ParamsType::eDynamicSceneBvhQuality,
            "(BuildQuality) Dynamic scene BVH build quality.")
        .def_rw(
            "dynamic_mesh_bvh_quality",
            &ParamsType::eDynamicMeshBvhQuality,
            "(BuildQuality) Dynamic mesh BVH build quality.")
        .def_rw(
            "static_scene_bvh_quality",
            &ParamsType::eStaticSceneBvhQuality,
            "(BuildQuality) Static scene BVH build quality.")
        .def_rw(
            "static_mesh_bvh_quality",
            &ParamsType::eStaticMeshBvhQuality,
            "(BuildQuality) Static mesh BVH build quality.")
        .def_rw(
            "max_vertex_face_contacts_estimate",
            &ParamsType::nMaxVertexFaceContactsEstimate,
            "(int) Max vertex-face contacts estimate for memory pre-allocation.")
        .def_rw(
            "max_face_vertex_contacts_estimate",
            &ParamsType::nMaxFaceVertexContactsEstimate,
            "(int) Max face-vertex contacts estimate for memory pre-allocation.")
        .def_rw(
            "max_edge_face_contacts_estimate",
            &ParamsType::nMaxEdgeFaceContactsEstimate,
            "(int) Max edge-face contacts estimate for memory pre-allocation.")
        .def_rw(
            "gammap",
            &ParamsType::gammap,
            "(float) Relaxation parameter for vertex displacement bound, must satisfy 0 < gammap < 0.5.")
        .def_rw(
            "gammae",
            &ParamsType::gammae,
            "(float) Proportion of bounds-violating vertices to trigger collision detection.")
        .def(
            "with_radii",
            &ParamsType::WithRadii,
            nb::arg("r"),
            nb::arg("rq"),
            nb::rv_policy::reference_internal,
            "Set contact and query radii.\n\n"
            "Args:\n"
            "    r (float): Contact radius.\n"
            "    rq (float): Query radius.\n\n"
            "Returns:\n"
            "    Params: Reference to this parameter set.")
        .def(
            "with_scene_features",
            &ParamsType::WithSceneFeatures,
            nb::arg("features"),
            nb::rv_policy::reference_internal,
            "Set scene features.\n\n"
            "Args:\n"
            "    features (SceneFeatures): Scene features.\n\n"
            "Returns:\n"
            "    Params: Reference to this parameter set.")
        .def(
            "with_dynamic_build_quality",
            &ParamsType::WithDynamicBuildQuality,
            nb::arg("scene"),
            nb::arg("mesh"),
            nb::rv_policy::reference_internal,
            "Set both dynamic scene and mesh BVH build quality.\n\n"
            "Args:\n"
            "    scene (BuildQuality): Build quality for dynamic scene.\n"
            "    mesh (BuildQuality): Build quality for dynamic mesh.\n\n"
            "Returns:\n"
            "    Params: Reference to this parameter set.")
        .def(
            "with_static_build_quality",
            &ParamsType::WithStaticBuildQuality,
            nb::arg("scene"),
            nb::arg("mesh"),
            nb::rv_policy::reference_internal,
            "Set both static scene and mesh BVH build quality.\n\n"
            "Args:\n"
            "    scene (BuildQuality): Build quality for static scene.\n"
            "    mesh (BuildQuality): Build quality for static mesh.\n\n"
            "Returns:\n"
            "    Params: Reference to this parameter set.")
        .def(
            "with_max_contact_estimates",
            &ParamsType::WithMaxContactEstimates,
            nb::arg("nvf"),
            nb::arg("nfv"),
            nb::arg("nef"),
            nb::rv_policy::reference_internal,
            "Set maximum number of contacts.\n\n"
            "Args:\n"
            "    nvf (int): Max vertex-face contacts estimate.\n"
            "    nfv (int): Max face-vertex contacts estimate.\n"
            "    nef (int): Max edge-face contacts estimate.\n\n"
            "Returns:\n"
            "    Params: Reference to this parameter set.")
        .def(
            "with_displacement_bound_config",
            &ParamsType::WithDisplacementBoundConfig,
            nb::arg("gammap"),
            nb::arg("gammae"),
            nb::rv_policy::reference_internal,
            "Set displacement bound parameters.\n\n"
            "Args:\n"
            "    gammap (float): Relaxation parameter for vertex displacement bound, must satisfy 0 < gammap < 0.5.\n"
            "    gammae (float): Proportion of bounds-violating vertices to trigger collision detection.\n\n"
            "Returns:\n"
            "    Params: Reference to this parameter set.")
        .def(
            "construct",
            &ParamsType::Construct,
            nb::arg("validate") = true,
            nb::rv_policy::reference_internal,
            "Validate and construct the parameters.\n\n"
            "Args:\n"
            "    validate (bool): Whether to validate parameters.\n\n"
            "Returns:\n"
            "    Params: Reference to this parameter set.")
        .def(
            "serialize",
            &ParamsType::Serialize,
            nb::arg("archive"),
            "Serialize the parameters to an archive.\n\n"
            "Args:\n"
            "    archive (pbat.io.Archive): Archive to serialize to.")
        .def(
            "deserialize",
            &ParamsType::Deserialize,
            nb::arg("archive"),
            "Deserialize the parameters from an archive.\n\n"
            "Args:\n"
            "    archive (pbat.io.Archive): Archive to deserialize from.");
}

} // namespace pbat::py::sim::contact::ogc