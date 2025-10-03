import meshio
import argparse
import pbatoolkit as pbat
import polyscope as ps
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Tetrahedral mesh partitioning viewer",
    )
    parser.add_argument("-i", "--input", help="Path to input mesh", type=str,
                        dest="input", required=True)
    parser.add_argument("-n", "--nparts", help="Number of partitions", type=int,
                        dest="nparts", default=8)
    args = parser.parse_args()

    imesh = meshio.read(args.input)
    V, C = imesh.points, imesh.cells_dict["tetra"]
    GGT = pbat.graph.mesh_dual_graph(C.T, V.shape[0])
    GGT.data[GGT.data < 3] = 0
    GGT.eliminate_zeros()
    weights = np.ones_like(GGT.data)
    partitions = pbat.graph.partition(
        GGT.indptr, GGT.indices, weights, args.nparts,
        # Default | MinEdgeCut | MinCommunicationVolume
        objective=pbat.graph.PartitioningObjective.MinEdgeCut,
        # Default | RandomMatching | SortedHeavyEdgeMatching
        coarsening=pbat.graph.PartitioningCoarseningStrategy.SortedHeavyEdgeMatching,
        # Default| GreedyBisectionGrowing | EdgeCutSeparator  | GreedyNodeBisectionGrowing | RandomBisectionAndRefinement
        initializer=pbat.graph.InitialPartitioningStrategy.GreedyBisectionGrowing,
        # Default | FiducciaMattheyses | GreedyCutAndVolumeRefinement | OneSidedNodeFiducciaMattheyses | TwoSidedNodeFiducciaMattheyses
        refinement=pbat.graph.PartitioningRefinementStrategy.TwoSidedNodeFiducciaMattheyses,
        n_partition_trials=10,
        n_separators=10,
        n_refinement_iters=100,
        seed=0,
        minimize_degree=True,
        with_two_hop=True,
        contiguous_parts=True,
        identify_conn_comp=False
    )
    ps.set_program_name("Mesh partitioning")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_SSAA_factor(4)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.init()
    vm = ps.register_volume_mesh("Mesh", V, C)
    vm.add_scalar_quantity("Partitioning", partitions,
                           defined_on="cells", cmap="turbo", enabled=True)
    ps.show()
