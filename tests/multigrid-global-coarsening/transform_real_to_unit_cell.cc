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
  GridGenerator::reference_cell(tria, reference_cell);

  Point<dim> p(1.0, 0.0);

  const Point<dim> p_cell =
    mapping.transform_real_to_unit_cell(tria.begin(), p);

  deallog << p_cell << std::endl;
}

main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  deallog.precision(8);

  test<2>();
}