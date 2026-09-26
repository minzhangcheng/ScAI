"""Process creation remains blocked through imports and alias assignment."""
import pytest

from backend.utils.sandbox import validate_code_safety


@pytest.mark.parametrize("code, expected_reason", [
    pytest.param(
        "import subprocess\nsubprocess.run(['pip', 'install', 'palettable'])", 'subprocess',
        id='subprocess_pip_install_is_blocked',
    ),
    pytest.param(
        "import os\nos.system('pip install palettable')", 'os.system',
        id='os_system_pip_install_is_blocked',
    ),
    pytest.param(
        "import subprocess\nprint('ready')", 'subprocess',
        id='plain_subprocess_import_is_blocked_before_cross_cell_aliasing',
    ),
    pytest.param(
        "from subprocess import run\nrun(['pip', 'install', 'palettable'])", 'subprocess',
        id='subprocess_from_import_run_pip_install_is_blocked',
    ),
    pytest.param(
        "import subprocess as sp\nsp.run(['pip', 'install', 'palettable'])", 'subprocess',
        id='subprocess_alias_pip_install_is_blocked',
    ),
    pytest.param(
        "from subprocess import run\nr = run\nr(['pip', 'install', 'palettable'])", 'subprocess',
        id='subprocess_from_import_assigned_run_is_blocked',
    ),
    pytest.param(
        "import subprocess as sp\nr = sp.run\nr(['pip', 'install', 'palettable'])", 'subprocess',
        id='subprocess_alias_assigned_run_is_blocked',
    ),
    pytest.param(
        "import subprocess as sp\nproc = sp\nproc.run(['pip', 'install', 'palettable'])", 'subprocess',
        id='subprocess_module_alias_assignment_is_blocked',
    ),
    pytest.param(
        "from subprocess import run\nr, other = run, None\nr(['pip', 'install', 'palettable'])", 'subprocess',
        id='subprocess_tuple_assigned_run_is_blocked',
    ),
    pytest.param(
        "from subprocess import run\nr: object = run\nr(['pip', 'install', 'palettable'])", 'subprocess',
        id='subprocess_annotated_assigned_run_is_blocked',
    ),
    pytest.param(
        "from os import system\nsystem('pip install palettable')", 'os.system',
        id='os_from_import_system_pip_install_is_blocked',
    ),
    pytest.param(
        "import os as operating_system\noperating_system.system('pip install palettable')", 'os.system',
        id='os_alias_system_pip_install_is_blocked',
    ),
    pytest.param(
        "from os import system\ns = system\ns('pip install palettable')", 'os.system',
        id='os_from_import_assigned_system_is_blocked',
    ),
    pytest.param(
        "import os as operating_system\ns = operating_system.system\ns('pip install palettable')", 'os.system',
        id='os_alias_assigned_system_is_blocked',
    ),
    pytest.param(
        "import os as operating_system\nops = operating_system\nops.system('pip install palettable')", 'os.system',
        id='os_module_alias_assignment_is_blocked',
    ),
    pytest.param(
        "from os import system\ns, other = system, None\ns('pip install palettable')", 'os.system',
        id='os_tuple_assigned_system_is_blocked',
    ),
    pytest.param(
        "import os as operating_system\nops: object = operating_system\nops.system('pip install palettable')", 'os.system',
        id='os_annotated_module_alias_assignment_is_blocked',
    ),
])
def test_process_creation_is_blocked(code, expected_reason):
    safe, reason = validate_code_safety(code)
    assert safe is False
    assert expected_reason in reason.lower().replace("os system", "os.system")
