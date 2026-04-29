@echo off
setlocal enabledelayedexpansion

REM === Configuration ===
set "INPUT_VIDEO=input\s4e3_out.mp4"
set "OUTPUT_VIDEO=output\output_48fps.mp4"

REM Directories for intermediate files
set "SCENE_DIR=output\scenes"
set "SPLIT_DIR=output\split_scenes"
set "FRAME_DIR=output\frames"
set "RIFE_INPUT=output\frames"
set "RIFE_OUTPUT=output\interpolated"
set "ENCODED_DIR=output\encoded_scenes"
set "CONCAT_LIST=output\concat_list.txt"
set "LOG_DIR=output\logs"

REM Paths to executables and models
set "RIFE_EXEC=libs\rife_ncnn_vulkan\rife-ncnn-vulkan.exe"
set "RIFE_MODEL=libs\rife_ncnn_vulkan\rife-v4.6"

REM Create necessary directories if they don't exist
for %%D in ("%SCENE_DIR%" "%SPLIT_DIR%" "%FRAME_DIR%" "%RIFE_OUTPUT%" "%ENCODED_DIR%" "%LOG_DIR%") do (
    if not exist "%%~D" (
        mkdir "%%~D"
        if !errorlevel! neq 0 (
            echo Error: Failed to create directory %%~D
            exit /b !errorlevel!
        )
    )
)

REM === Step 1 and 2: Combined Downscaling and Scene Detection ===
echo [Step 1 and 2] Downscaling and detecting scene changes...

ffmpeg -y -i "%INPUT_VIDEO%" -vf "scale=iw/12:ih/12,select='gt(scene,0.4)',showinfo" -f null - 2> "%LOG_DIR%\scenes.log"

if %errorlevel% neq 0 (
    echo Error: Downscaling and scene detection failed. Check scenes.log for details.
    exit /b %errorlevel%
)

echo Downscaling and scene detection completed.

REM === Extract Scene Change Timestamps ===
echo Extracting scene change timestamps...

REM Initialize timestamp array
set "count=0"
set "timestamps[0]=0"

REM Extract pts_time using findstr and for loop
for /f "tokens=*" %%a in ('findstr /C:"pts_time:" "%LOG_DIR%\scenes.log"') do (
    for /f "tokens=2 delims=:" %%b in ("%%a") do (
        set "timestamp=%%b"
    )
    REM Remove any whitespace
    set "timestamp=!timestamp: =!"
    REM Ensure the timestamp is numeric
    set "isNumeric=1"
    for /f "delims=0123456789." %%c in ("!timestamp!") do (
        set "isNumeric=0"
    )
    if "!isNumeric!"=="1" (
        echo Detected timestamp: !timestamp!
        set /a count+=1
        set "timestamps[!count!]=!timestamp!"
    ) else (
        echo Warning: Invalid timestamp detected and skipped: !timestamp!
    )
)

echo Total number of detected scenes: %count%

REM Get total duration of the original video using ffprobe
echo Retrieving total video duration using ffprobe...
set "total_duration=27.237667"

echo Total duration of video: %total_duration%

REM Add the total duration as the last timestamp
set /a end_index=count+1
set "timestamps[!end_index!]=%total_duration%"

REM === Step 3: Splitting Original Video into Scenes ===
echo [Step 3] Splitting original video into scenes...

REM Split the original high-resolution video into scenes based on timestamps
for /l %%i in (0,1,%count%) do (
    set "start=!timestamps[%%i]!"
    set /a next=%%i+1
    set "end=!timestamps[!next!]!"
    if "!end!"=="" (
        set "end=%total_duration%"
    )
    echo Splitting scene %%i: Start=!start! | End=!end!
    ffmpeg -y -i "%INPUT_VIDEO%" -ss !start! -to !end! -c copy "%SPLIT_DIR%\scene_%%i.mp4" > "%LOG_DIR%\split_scene_%%i.log" 2>&1
    if !errorlevel! neq 0 (
        echo Error: Splitting scene %%i failed. Check split_scene_%%i.log for details.
        exit /b !errorlevel!
    )
)

echo Scene splitting completed.

REM === Step 4: Frame Extraction ===
echo [Step 4] Extracting frames from each scene...

REM Extract frames from each split scene using FFmpeg
for %%s in ("%SPLIT_DIR%\scene_*.mp4") do (
    set "scene_name=%%~ns"
    if not exist "%FRAME_DIR%\!scene_name!" (
        mkdir "%FRAME_DIR%\!scene_name!"
        if !errorlevel! neq 0 (
            echo Error: Failed to create frame directory for !scene_name!.
            exit /b !errorlevel!
        )
    )
    echo Extracting frames for !scene_name!...
    ffmpeg -y -i "%%s" -vsync vfr -q:v 2 "%FRAME_DIR%\!scene_name!\frame_%%08d.jpg" > "%LOG_DIR%\extract_!scene_name!.log" 2>&1
    if !errorlevel! neq 0 (
        echo Error: Frame extraction for !scene_name! failed. Check extract_!scene_name!.log for details.
        exit /b !errorlevel!
    )
)

echo Frame extraction completed.

REM === Step 5: Frame Interpolation with RIFE ===
echo [Step 5] Applying frame interpolation with RIFE...

REM Process each scene's frames with RIFE sequentially to handle logging correctly
for /d %%d in ("%RIFE_INPUT%\scene_*") do (
    set "scene_dir=%%~nd"
    if not exist "%RIFE_OUTPUT%\!scene_dir!" (
        mkdir "%RIFE_OUTPUT%\!scene_dir!"
        if !errorlevel! neq 0 (
            echo Error: Failed to create interpolated directory for !scene_dir!.
            exit /b !errorlevel!
        )
    )
    echo Interpolating frames for !scene_dir!...
    "%RIFE_EXEC%" -i "%%d" -o "%RIFE_OUTPUT%\!scene_dir!" -m "%RIFE_MODEL%" -v -g 0 -j 8:32:8 -f "frame_%%08d.jpg" > "%LOG_DIR%\rife_!scene_dir!.log" 2>&1
    if !errorlevel! neq 0 (
        echo Error: RIFE interpolation for !scene_dir! failed. Check rife_!scene_dir!.log for details.
        exit /b !errorlevel!
    )
)

echo Frame interpolation completed.

REM === Step 6: Re-encoding Each Scene ===
echo [Step 6] Re-encoding interpolated frames into video...

REM Re-encode each interpolated scene using NVENC
for %%s in ("%RIFE_OUTPUT%\scene_*") do (
    set "scene_name=%%~ns"
    echo Re-encoding !scene_name!...
    ffmpeg -y -framerate 47.96 -i "%%s\frame_%%08d.png" -c:v hevc_nvenc -preset p7 -rc constqp -qp 16 -pix_fmt yuv420p "%ENCODED_DIR%\!scene_name!.mp4" > "%LOG_DIR%\encode_!scene_name!.log" 2>&1
    if !errorlevel! neq 0 (
        echo Error: Encoding for !scene_name! failed. Check encode_!scene_name!.log for details.
        exit /b !errorlevel!
    )
)

echo Re-encoding completed.

REM === Step 7: Concatenating All Encoded Scenes ===
echo [Step 7] Concatenating all encoded scenes into the final video...

REM Ensure CONCAT_LIST is empty before writing
> "%CONCAT_LIST%" (
    for /f "delims=" %%e in ('dir /b /on "%ENCODED_DIR%\scene_*.mp4"') do (
        echo file '%%e'
    )
)

REM Change directory to ENCODED_DIR to simplify paths in CONCAT_LIST
pushd "%ENCODED_DIR%"
ffmpeg -y -f concat -safe 0 -i "%CONCAT_LIST%" -c copy "%~dp0%OUTPUT_VIDEO%" > "%LOG_DIR%\concatenate.log" 2>&1
popd

if %errorlevel% neq 0 (
    echo Error: Concatenation failed. Check concatenate.log for details.
    exit /b %errorlevel%
)

echo Concatenation completed. Final video is "%OUTPUT_VIDEO%"

echo [Process Completed Successfully]
endlocal
pause
