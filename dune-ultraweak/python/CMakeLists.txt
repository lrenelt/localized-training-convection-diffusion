dune_symlink_to_source_files(FILES compute_example_solutions.py)
dune_symlink_to_source_files(FILES run_all_greedy_tests.py)
dune_symlink_to_source_files(FILES test_norms.py)

configure_file(pyenv.sh.in pyenv.sh)

add_subdirectory(common)
add_subdirectory(convectionDiffusion)
