@echo off
rem Build llama-server (CUDA) for the Relay harness.
rem Usage: scripts\build-llama-server.bat
rem Env overrides: VSVARS (VsDevCmd.bat path), CUDAToolkit_ROOT, CMAKE_BIN, LLAMA_SRC_REF
setlocal
set REPO=%~dp0..
if not defined VSVARS set "VSVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
if not defined CUDAToolkit_ROOT set "CUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
rem Pin the whole session to the selected toolkit: a system-wide CUDA_PATH
rem pointing at a different version mixes nvcc and link libraries (LNK2005).
set "CUDA_PATH=%CUDAToolkit_ROOT%"
set "CUDA_PATH_V13_2=%CUDAToolkit_ROOT%"
if not defined CMAKE_BIN set "CMAKE_BIN=%REPO%\.venv\Scripts\cmake.exe"
if not defined LLAMA_SRC_REF set "LLAMA_SRC_REF="

if not exist "%VSVARS%" (
  echo Missing VS Build Tools. Set VSVARS to your VsDevCmd.bat path.
  exit /b 1
)
call "%VSVARS%" -arch=amd64 >nul
if errorlevel 1 (
  echo VsDevCmd failed.
  exit /b 1
)

if not exist "%REPO%\third_party\llama.cpp\CMakeLists.txt" (
  echo Cloning llama.cpp into third_party\llama.cpp ...
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "%REPO%\third_party\llama.cpp"
  if errorlevel 1 exit /b 1
)
if not "%LLAMA_SRC_REF%"=="" (
  git -C "%REPO%\third_party\llama.cpp" fetch --depth 1 origin %LLAMA_SRC_REF%
  git -C "%REPO%\third_party\llama.cpp" checkout %LLAMA_SRC_REF%
)

"%CMAKE_BIN%" -S "%REPO%\third_party\llama.cpp" -B "%REPO%\third_party\llama.cpp\build" ^
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON ^
  -DCUDAToolkit_ROOT="%CUDAToolkit_ROOT%" ^
  -DCMAKE_CUDA_COMPILER="%CUDAToolkit_ROOT%\bin\nvcc.exe"
if errorlevel 1 exit /b 1

"%CMAKE_BIN%" --build "%REPO%\third_party\llama.cpp\build" --target llama-server
if errorlevel 1 exit /b 1

echo Built: %REPO%\third_party\llama.cpp\build\bin\llama-server.exe
