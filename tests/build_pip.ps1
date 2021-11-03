$ErrorActionPreference = "Stop"

if (test-path _skbuild) {
    cmd.exe /c rd /s /q _skbuild
}
if (test-path dist) {
    cmd.exe /c rd /s /q dist
}
if (test-path ..\venv_ngs) {
    cmd.exe /c rd /s /q ..\venv_ngs
}


$env:NETGEN_CCACHE = 1

$py=$args[0]
$netgen_version=(& $py\python.exe tests\get_python_version_string_from_git.py external_dependencies\netgen)

& $py\python.exe -m venv ..\venv_ngs
..\venv_ngs\scripts\Activate.ps1
$env:PATH += ";$env:CI_PROJECT_DIR\venv\bin"
python --version

pip3 install scikit-build wheel numpy twine mkl-devel mkl
pip3 install -i https://test.pypi.org/simple/ netgen-mesher==$netgen_version
python setup.py bdist_wheel -G"Visual Studio 16 2019"
python -m twine upload --repository testpypi dist\*.whl