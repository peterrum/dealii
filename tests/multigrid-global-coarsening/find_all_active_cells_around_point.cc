#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include "../tests.h"

template <int dim>
void
test()
{
  const auto  reference_cell = ReferenceCells::get_simplex<dim>();
  const auto &mapping =
    reference_cell.template get_default_linear_mapping<dim>();

  Triangulation<dim> tria;
  if (false)
    GridGenerator::reference_cell(tria, reference_cell);
  else
    GridGenerator::subdivided_hyper_cube_with_simplices(tria, 1);

  GridTools::Cache<dim> cache(tria, mapping);

  Point<dim> p(0.5, 0.5);

  const auto result = GridTools::find_all_active_cells_around_point(
    mapping, tria, p, 1e-6, {tria.begin(), p});

  deallog << result.size() << std::endl; // should be 2

  // expectation: each cell finds 6 points with reference positions
  // (0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 0.5), (0.5, 0.5), (0.0, 1.0)
}

main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  deallog.precision(8);

  test<2>();
}