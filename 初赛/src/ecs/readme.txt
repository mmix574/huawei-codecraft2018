WSL 运行本地测试
python ecs.py TrainData_2015.1.1_2015.2.19.txt input_5flavors_cpu_7days.txt output.txt
python ecs.py TrainData_2015.1.1_2015.2.19.txt input_15flavors_cpu_7days.txt output.txt
python ecs.py TrainData_2015.1.1_2015.2.19.txt input_15flavors_cpu_7days.txt output.txt


运行打包程序
tar -czvf my.tar.gz  src/

最新打包：
tar -czvf -6day_5.tar.gz src/ecs/*.py src/ecs/*.txt src/ecs/learn/*.py src/ecs/ecs.py src/ecs/linalg/*.py src/ecs/ecs.py src/ecs/predictions/*.py 


运行批量评测
python evaluate.py train 5
python evaluate.py test
python evaluate.py val

