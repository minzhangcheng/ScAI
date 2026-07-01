from backend.utils.sandbox import validate_code_safety


def test_subprocess_pip_install_is_blocked():
    safe, reason = validate_code_safety(
        "import subprocess\nsubprocess.run(['pip', 'install', 'palettable'])"
    )

    assert safe is False
    assert "subprocess" in reason.lower()


def test_os_system_pip_install_is_blocked():
    safe, reason = validate_code_safety(
        "import os\nos.system('pip install palettable')"
    )

    assert safe is False
    assert "os.system" in reason.lower()


def test_plain_subprocess_import_without_call_remains_allowed():
    safe, reason = validate_code_safety("import subprocess\nprint('ready')")

    assert safe is True
    assert reason == ""


def test_subprocess_from_import_run_pip_install_is_blocked():
    safe, reason = validate_code_safety(
        "from subprocess import run\nrun(['pip', 'install', 'palettable'])"
    )

    assert safe is False
    assert "subprocess" in reason.lower()


def test_subprocess_alias_pip_install_is_blocked():
    safe, reason = validate_code_safety(
        "import subprocess as sp\nsp.run(['pip', 'install', 'palettable'])"
    )

    assert safe is False
    assert "subprocess" in reason.lower()


def test_subprocess_from_import_assigned_run_is_blocked():
    safe, reason = validate_code_safety(
        "from subprocess import run\nr = run\nr(['pip', 'install', 'palettable'])"
    )

    assert safe is False
    assert "subprocess" in reason.lower()


def test_subprocess_alias_assigned_run_is_blocked():
    safe, reason = validate_code_safety(
        "import subprocess as sp\nr = sp.run\nr(['pip', 'install', 'palettable'])"
    )

    assert safe is False
    assert "subprocess" in reason.lower()


def test_subprocess_module_alias_assignment_is_blocked():
    safe, reason = validate_code_safety(
        "import subprocess as sp\nproc = sp\nproc.run(['pip', 'install', 'palettable'])"
    )

    assert safe is False
    assert "subprocess" in reason.lower()


def test_subprocess_tuple_assigned_run_is_blocked():
    safe, reason = validate_code_safety(
        "from subprocess import run\nr, other = run, None\nr(['pip', 'install', 'palettable'])"
    )

    assert safe is False
    assert "subprocess" in reason.lower()


def test_subprocess_annotated_assigned_run_is_blocked():
    safe, reason = validate_code_safety(
        "from subprocess import run\nr: object = run\nr(['pip', 'install', 'palettable'])"
    )

    assert safe is False
    assert "subprocess" in reason.lower()


def test_os_from_import_system_pip_install_is_blocked():
    safe, reason = validate_code_safety(
        "from os import system\nsystem('pip install palettable')"
    )

    assert safe is False
    assert "os.system" in reason.lower() or "os system" in reason.lower()


def test_os_alias_system_pip_install_is_blocked():
    safe, reason = validate_code_safety(
        "import os as operating_system\noperating_system.system('pip install palettable')"
    )

    assert safe is False
    assert "os.system" in reason.lower() or "os system" in reason.lower()


def test_os_from_import_assigned_system_is_blocked():
    safe, reason = validate_code_safety(
        "from os import system\ns = system\ns('pip install palettable')"
    )

    assert safe is False
    assert "os.system" in reason.lower() or "os system" in reason.lower()


def test_os_alias_assigned_system_is_blocked():
    safe, reason = validate_code_safety(
        "import os as operating_system\ns = operating_system.system\ns('pip install palettable')"
    )

    assert safe is False
    assert "os.system" in reason.lower() or "os system" in reason.lower()


def test_os_module_alias_assignment_is_blocked():
    safe, reason = validate_code_safety(
        "import os as operating_system\nops = operating_system\nops.system('pip install palettable')"
    )

    assert safe is False
    assert "os.system" in reason.lower() or "os system" in reason.lower()


def test_os_tuple_assigned_system_is_blocked():
    safe, reason = validate_code_safety(
        "from os import system\ns, other = system, None\ns('pip install palettable')"
    )

    assert safe is False
    assert "os.system" in reason.lower() or "os system" in reason.lower()


def test_os_annotated_module_alias_assignment_is_blocked():
    safe, reason = validate_code_safety(
        "import os as operating_system\nops: object = operating_system\nops.system('pip install palettable')"
    )

    assert safe is False
    assert "os.system" in reason.lower() or "os system" in reason.lower()
