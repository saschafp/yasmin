import yasmin.analysis.system as system


def test_available_cores_honors_omp_num_threads(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setattr(
        system.os,
        "sched_getaffinity",
        lambda _pid: set(range(72)),
        raising=False,
    )

    assert system.available_cores() == 4
