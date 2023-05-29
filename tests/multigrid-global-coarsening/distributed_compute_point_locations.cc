#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include "../tests.h"

template <int dim>
void
test()
{
  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube_with_simplices(tria, 1);
  tria.refine_global();

  const auto &mapping = ReferenceCells::get_simplex<dim>()
                          .template get_default_linear_mapping<dim>();

  std::vector<Point<dim>> positions;

  const unsigned int n = 4;

  for (unsigned int i = 0; i <= n; ++i)
    for (unsigned int j = 0; j <= n; ++j)
      positions.emplace_back(1.0 / n * i, 1.0 / n * j);

  GridTools::Cache<dim> cache(tria, mapping);

  std::vector<std::vector<BoundingBox<dim>>> global_bounding_boxes(1);
  global_bounding_boxes[0].push_back(
    GridTools::compute_bounding_box(tria).create_extended(1e-6));

  const auto cells_positions_and_index_maps =
    GridTools::internal::distributed_compute_point_locations(
      cache,
      positions,
      global_bounding_boxes,
      std::vector<bool>(),
      1e-6,
      true,
      false);
}

main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  deallog.precision(8);

  test<2>();
}