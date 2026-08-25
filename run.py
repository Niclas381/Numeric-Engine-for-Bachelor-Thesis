import mode_solver as ms

ms.save_sol(
    filename="mode_solutions.h5",
    run_name="test",
    k_values = [i for i in range(1, 11)],
    eta_values= [i for i in range(1, 11)],
    m = 1,
    x = 1,
    a_model_name="exponential(eta)",
    method="DOP853",
    rtol=1e-10,
    atol=1e-12
)