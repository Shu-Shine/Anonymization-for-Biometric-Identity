@echo off
echo Running script1...
C:\Users\you\Anaconda3\envs\myenv\python.exe filter_latest.py || echo filter_latest.py failed!

echo Running script2...
C:\Users\you\Anaconda3\envs\myenv\python.exe filter_latest.py || echo filter_latest.py failed!


echo All done!
pause
