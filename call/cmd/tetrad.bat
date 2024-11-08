
REM Run Tetrad

SET PREFIX=%~1
SET VERSION=%~2
SET PARAMS=%~3

cd call\java
java -jar causal-cmd-%VERSION%-jar-with-dependencies.jar %PARAMS% --dataset tmp/%PREFIX%.csv --delimiter comma --out tmp --prefix %PREFIX% --verbose
cd ..\..