def test_nonempty_model_empty_guide_ok(Elbo, strict_enumeration_warning):
def test_empty_model_empty_guide_ok(Elbo, strict_enumeration_warning):
def test_variable_clash_in_model_error(Elbo):
def test_model_guide_dim_mismatch_error(Elbo):
def test_model_guide_shape_mismatch_error(Elbo):
def test_variable_clash_in_guide_error(Elbo):
def test_iplate_ok(subsample_size, Elbo):
def test_iplate_variable_clash_error(Elbo):
def test_plate_ok(subsample_size, Elbo):
def test_plate_no_size_ok(Elbo):
def test_iplate_iplate_ok(subsample_size, Elbo, max_plate_nesting):
def test_iplate_iplate_swap_ok(subsample_size, Elbo, max_plate_nesting):
def test_iplate_in_model_not_guide_ok(subsample_size, Elbo):
def test_iplate_in_guide_not_model_error(subsample_size, Elbo, is_validate):
def test_plate_broadcast_error(Elbo):
def test_plate_iplate_ok(Elbo):
def test_iplate_plate_ok(Elbo):
def test_nested_plate_plate_ok(Elbo):
def test_plate_reuse_ok(Elbo):
def test_nested_plate_plate_dim_error_1(Elbo):
def test_nested_plate_plate_dim_error_2(Elbo):
def test_nested_plate_plate_dim_error_3(Elbo):
def test_nested_plate_plate_dim_error_4(Elbo):
def test_nonnested_plate_plate_ok(Elbo):
def test_three_indep_plate_at_different_depths_ok():
def test_plate_wrong_size_error():
def test_enum_discrete_misuse_warning(Elbo, enumerate_):
#######################
def test_enum_discrete_single_ok():
def test_enum_discrete_missing_config_warning(strict_enumeration_warning):
def test_enum_discrete_single_single_ok():
def test_enum_discrete_iplate_single_ok():
def test_plate_enum_discrete_batch_ok():
def test_plate_enum_discrete_no_discrete_vars_warning(strict_enumeration_warning):
def test_no_plate_enum_discrete_batch_error():
def test_enum_discrete_parallel_ok(max_plate_nesting):
def test_enum_discrete_parallel_nested_ok(max_plate_nesting):
def test_enumerate_parallel_plate_ok(enumerate_, expand, num_samples):
def test_enum_discrete_plate_dependency_warning(enumerate_, is_validate, max_plate_nesting):
def test_enum_discrete_iplate_plate_dependency_ok(enumerate_, max_plate_nesting):
def test_enum_discrete_iplates_plate_dependency_warning(enumerate_, is_validate, max_plate_nesting)
def test_enum_discrete_plates_dependency_ok(enumerate_):
def test_enum_discrete_non_enumerated_plate_ok(enumerate_):
# wy: Trace_ELBO
def test_plate_shape_broadcasting():
def test_enum_discrete_plate_shape_broadcasting_ok(enumerate_, expand, num_samples):
# wy: Trace_ELBO
def test_dim_allocation_ok(Elbo, expand):
# wy: Trace_ELBO
def test_dim_allocation_error(Elbo, expand):
def test_enum_in_model_ok():
def test_enum_in_model_plate_ok():
def test_enum_sequential_in_model_error():
def test_enum_in_model_plate_reuse_ok():
def test_enum_in_model_multi_scale_error():
def test_enum_in_model_diamond_error():
# wy: Trace_ELBO
def test_vectorized_num_particles(Elbo):
def test_enum_discrete_vectorized_num_particles(enumerate_, expand, num_samples, num_particles):
def test_enum_recycling_chain():
def test_enum_recycling_dbn(markov):
def test_enum_recycling_nested():
def test_enum_recycling_grid():
def test_enum_recycling_reentrant():
def test_enum_recycling_reentrant_history(history):
def test_enum_recycling_mutual_recursion():
def test_enum_recycling_interleave():
def test_enum_recycling_plate():
def test_markov_history(history):
# wy: mf
def test_mean_field_ok():
# wy: mf
def test_mean_field_warn():
