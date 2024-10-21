[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YFgwt0yY)
# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2/module2/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py project/run_manual.py project/run_scalar.py project/datasets.py

## Task 2.5

### Task 2.5.1: Simple Dataset
50 data points

**Hyperparameters**:
- learning rate: 0.1
- max epochs: 1500
- hidden layers: 3

<img src="images\task2_5\1. simple\1. Dataset.png" width="50%">
<img src="images\task2_5\1. simple\2. Hidden Layer and Initial Setting.png" width="50%">
<img src="images\task2_5\1. simple\3. Hyperparameters and Results.png" width="50%">
<img src="images\task2_5\1. simple\4. Loss Graph and Table.png" width="50%">

**Simple Training Log**:
```
Epoch: 0/1500, loss: 0, correct: 0
Epoch: 10/1500, loss: 40.92874139085717, correct: 21
Epoch: 20/1500, loss: 36.668399506233676, correct: 21
Epoch: 30/1500, loss: 35.080167132876674, correct: 21
Epoch: 40/1500, loss: 34.370077759715315, correct: 30
Epoch: 50/1500, loss: 33.98290869320782, correct: 30
Epoch: 60/1500, loss: 33.68875576264421, correct: 29
Epoch: 70/1500, loss: 33.48120119222591, correct: 29
Epoch: 80/1500, loss: 33.27771168486874, correct: 29
Epoch: 90/1500, loss: 33.081317830114656, correct: 29
Epoch: 100/1500, loss: 32.87756623628576, correct: 29
Epoch: 110/1500, loss: 32.64640253031508, correct: 29
Epoch: 120/1500, loss: 32.37463585776924, correct: 32
Epoch: 130/1500, loss: 32.04822020086916, correct: 32
Epoch: 140/1500, loss: 31.66666321825208, correct: 33
Epoch: 150/1500, loss: 31.214456122798172, correct: 35
Epoch: 160/1500, loss: 30.712899590848284, correct: 38
Epoch: 170/1500, loss: 30.131428921745485, correct: 40
Epoch: 180/1500, loss: 29.391257347549303, correct: 42
Epoch: 190/1500, loss: 28.558700075271414, correct: 43
Epoch: 200/1500, loss: 27.68417281623093, correct: 43
Epoch: 210/1500, loss: 26.696916761012933, correct: 43
Epoch: 220/1500, loss: 25.66650389816633, correct: 43
Epoch: 230/1500, loss: 24.57476419024355, correct: 43
Epoch: 240/1500, loss: 23.43785728190349, correct: 45
Epoch: 250/1500, loss: 22.262770236689203, correct: 45
Epoch: 260/1500, loss: 21.10102697152695, correct: 47
Epoch: 270/1500, loss: 20.00006047705357, correct: 47
Epoch: 280/1500, loss: 18.909711001246173, correct: 47
Epoch: 290/1500, loss: 17.865216810307423, correct: 47
Epoch: 300/1500, loss: 16.88643688894778, correct: 47
Epoch: 310/1500, loss: 15.983637239967523, correct: 47
Epoch: 320/1500, loss: 15.136998858245923, correct: 48
Epoch: 330/1500, loss: 14.335349614243754, correct: 48
Epoch: 340/1500, loss: 13.583387344008846, correct: 48
Epoch: 350/1500, loss: 12.882666889248156, correct: 48
Epoch: 360/1500, loss: 12.230144698150536, correct: 49
Epoch: 370/1500, loss: 11.62913244102036, correct: 49
Epoch: 380/1500, loss: 11.08401977216936, correct: 49
Epoch: 390/1500, loss: 10.581330767186332, correct: 49
Epoch: 400/1500, loss: 10.117245027450135, correct: 49
Epoch: 410/1500, loss: 9.684712308519611, correct: 49
Epoch: 420/1500, loss: 9.283995661598553, correct: 49
Epoch: 430/1500, loss: 8.911582273860418, correct: 49
Epoch: 440/1500, loss: 8.570737279095152, correct: 49
Epoch: 450/1500, loss: 8.253705221698604, correct: 49
Epoch: 460/1500, loss: 7.957609634343393, correct: 49
Epoch: 470/1500, loss: 7.679009490946604, correct: 49
Epoch: 480/1500, loss: 7.416643890938518, correct: 49
Epoch: 490/1500, loss: 7.169357740291615, correct: 49
Epoch: 500/1500, loss: 6.936887482938311, correct: 49
Epoch: 510/1500, loss: 6.719107376994416, correct: 49
Epoch: 520/1500, loss: 6.515014672651822, correct: 49
Epoch: 530/1500, loss: 6.3237979307131145, correct: 49
Epoch: 540/1500, loss: 6.14340057214952, correct: 49
Epoch: 550/1500, loss: 5.9737137687343385, correct: 49
Epoch: 560/1500, loss: 5.813057834787351, correct: 49
Epoch: 570/1500, loss: 5.660533945923974, correct: 49
Epoch: 580/1500, loss: 5.515566778658576, correct: 49
Epoch: 590/1500, loss: 5.377636064329501, correct: 49
Epoch: 600/1500, loss: 5.246269170664238, correct: 49
Epoch: 610/1500, loss: 5.121035041713909, correct: 49
Epoch: 620/1500, loss: 5.001539186862925, correct: 49
Epoch: 630/1500, loss: 4.887621298508172, correct: 49
Epoch: 640/1500, loss: 4.77944094003338, correct: 49
Epoch: 650/1500, loss: 4.6761261382474535, correct: 49
Epoch: 660/1500, loss: 4.5771008501380015, correct: 49
Epoch: 670/1500, loss: 4.482123099101817, correct: 49
Epoch: 680/1500, loss: 4.3909673760116945, correct: 49
Epoch: 690/1500, loss: 4.303423259830966, correct: 49
Epoch: 700/1500, loss: 4.219294191061203, correct: 49
Epoch: 710/1500, loss: 4.139101329435461, correct: 49
Epoch: 720/1500, loss: 4.062411552022356, correct: 49
Epoch: 730/1500, loss: 3.9892662002888875, correct: 49
Epoch: 740/1500, loss: 3.918940461277286, correct: 49
Epoch: 750/1500, loss: 3.8511183341790427, correct: 49
Epoch: 760/1500, loss: 3.7856677801134673, correct: 49
Epoch: 770/1500, loss: 3.7232795817919087, correct: 49
Epoch: 780/1500, loss: 3.662864999093158, correct: 49
Epoch: 790/1500, loss: 3.6051100685286466, correct: 49
Epoch: 800/1500, loss: 3.5494653643304908, correct: 49
Epoch: 810/1500, loss: 3.4958514393471165, correct: 49
Epoch: 820/1500, loss: 3.4440868209460636, correct: 49
Epoch: 830/1500, loss: 3.394190252746441, correct: 49
Epoch: 840/1500, loss: 3.34584888371701, correct: 49
Epoch: 850/1500, loss: 3.2989655087745517, correct: 49
Epoch: 860/1500, loss: 3.253455433532994, correct: 49
Epoch: 870/1500, loss: 3.2092435915996926, correct: 49
Epoch: 880/1500, loss: 3.166262663278992, correct: 49
Epoch: 890/1500, loss: 3.12445166976957, correct: 50
Epoch: 900/1500, loss: 3.083754909798656, correct: 50
Epoch: 910/1500, loss: 3.044121144912694, correct: 50
Epoch: 920/1500, loss: 3.005502966835301, correct: 50
Epoch: 930/1500, loss: 2.967856299278991, correct: 50
Epoch: 940/1500, loss: 2.931207721200805, correct: 50
Epoch: 950/1500, loss: 2.895453636224876, correct: 50
Epoch: 960/1500, loss: 2.860523099132619, correct: 50
Epoch: 970/1500, loss: 2.8267151057517172, correct: 50
Epoch: 980/1500, loss: 2.793142105174388, correct: 50
Epoch: 990/1500, loss: 2.760700835784148, correct: 50
Epoch: 1000/1500, loss: 2.7290149879234074, correct: 50
Epoch: 1010/1500, loss: 2.697956245588318, correct: 50
Epoch: 1020/1500, loss: 2.6672435526393072, correct: 50
Epoch: 1030/1500, loss: 2.637482129697294, correct: 50
Epoch: 1040/1500, loss: 2.6083911674979166, correct: 50
Epoch: 1050/1500, loss: 2.5800278591894417, correct: 50
Epoch: 1060/1500, loss: 2.5522480763799864, correct: 50
Epoch: 1070/1500, loss: 2.525021951750834, correct: 50
Epoch: 1080/1500, loss: 2.4983312208255386, correct: 50
Epoch: 1090/1500, loss: 2.472158469062775, correct: 50
Epoch: 1100/1500, loss: 2.44648708359668, correct: 50
Epoch: 1110/1500, loss: 2.4213012053423384, correct: 50
Epoch: 1120/1500, loss: 2.396585684868841, correct: 50
Epoch: 1130/1500, loss: 2.37232604163644, correct: 50
Epoch: 1140/1500, loss: 2.348490381391333, correct: 50
Epoch: 1150/1500, loss: 2.3248474777899966, correct: 50
Epoch: 1160/1500, loss: 2.3019853961604175, correct: 50
Epoch: 1170/1500, loss: 2.279490500176875, correct: 50
Epoch: 1180/1500, loss: 2.2571202486234236, correct: 50
Epoch: 1190/1500, loss: 2.235369385625796, correct: 50
Epoch: 1200/1500, loss: 2.214097321347944, correct: 50
Epoch: 1210/1500, loss: 2.1930255798277902, correct: 50
Epoch: 1220/1500, loss: 2.1721206213091464, correct: 50
Epoch: 1230/1500, loss: 2.1519481432404133, correct: 50
Epoch: 1240/1500, loss: 2.131747323298521, correct: 50
Epoch: 1250/1500, loss: 2.1120218188220496, correct: 50
Epoch: 1260/1500, loss: 2.0926505493057594, correct: 50
Epoch: 1270/1500, loss: 2.0735710390267883, correct: 50
Epoch: 1280/1500, loss: 2.0547732055062373, correct: 50
Epoch: 1290/1500, loss: 2.036292577839742, correct: 50
Epoch: 1300/1500, loss: 2.018100304354049, correct: 50
Epoch: 1310/1500, loss: 2.000188191522027, correct: 50
Epoch: 1320/1500, loss: 1.9825485825955362, correct: 50
Epoch: 1330/1500, loss: 1.9651742772619096, correct: 50
Epoch: 1340/1500, loss: 1.9480584683776787, correct: 50
Epoch: 1350/1500, loss: 1.9311946918988372, correct: 50
Epoch: 1360/1500, loss: 1.9145767869940735, correct: 50
Epoch: 1370/1500, loss: 1.8981988640109886, correct: 50
Epoch: 1380/1500, loss: 1.8820552785001805, correct: 50
Epoch: 1390/1500, loss: 1.8661406099181486, correct: 50
Epoch: 1400/1500, loss: 1.8504496439523865, correct: 50
Epoch: 1410/1500, loss: 1.8349773576602764, correct: 50
Epoch: 1420/1500, loss: 1.819718906804117, correct: 50
Epoch: 1430/1500, loss: 1.804669614910803, correct: 50
Epoch: 1440/1500, loss: 1.7898249636959385, correct: 50
Epoch: 1450/1500, loss: 1.7751805845772748, correct: 50
Epoch: 1460/1500, loss: 1.760732251066851, correct: 50
Epoch: 1470/1500, loss: 1.7464758718802422, correct: 50
Epoch: 1480/1500, loss: 1.7324074846386017, correct: 50
Epoch: 1490/1500, loss: 1.7185232500671954, correct: 50
Epoch: 1500/1500, loss: 1.7048194466158382, correct: 50
```

### Task 2.5.2: Diag Dataset
50 data points

**Hyperparameters**:
- learning rate: 0.1
- max epochs: 1500
- hidden layers: 3

<img src="images\task2_5\2. diag\1. Dataset.png" width="50%">
<img src="images\task2_5\2. diag\2. Hidden Layer and Initial Setting.png" width="50%">
<img src="images\task2_5\2. diag\3. Hyperparameters and Results.png" width="50%">
<img src="images\task2_5\2. diag\4. Loss Graph and Table.png" width="50%">

**Diag Training Log**:
```
Epoch: 0/1500, loss: 0, correct: 0
Epoch: 0/1500, loss: 0, correct: 0
Epoch: 10/1500, loss: 15.655589636356044, correct: 46
Epoch: 20/1500, loss: 15.04421671840529, correct: 46
Epoch: 30/1500, loss: 14.719896037113568, correct: 46
Epoch: 40/1500, loss: 14.53295109327772, correct: 46
Epoch: 50/1500, loss: 14.411809176799437, correct: 46
Epoch: 60/1500, loss: 14.323463996910666, correct: 46
Epoch: 70/1500, loss: 14.252458674014996, correct: 46
Epoch: 80/1500, loss: 14.19138648266034, correct: 46
Epoch: 90/1500, loss: 14.136537950438525, correct: 46
Epoch: 100/1500, loss: 14.085932000503622, correct: 46
Epoch: 110/1500, loss: 14.038412816154647, correct: 46
Epoch: 120/1500, loss: 13.993231784337528, correct: 46
Epoch: 130/1500, loss: 13.949851423293827, correct: 46
Epoch: 140/1500, loss: 13.907833813926656, correct: 46
Epoch: 150/1500, loss: 13.866826256396225, correct: 46
Epoch: 160/1500, loss: 13.82649853104555, correct: 46
Epoch: 170/1500, loss: 13.786571704434902, correct: 46
Epoch: 180/1500, loss: 13.746789815097411, correct: 46
Epoch: 190/1500, loss: 13.706861267367831, correct: 46
Epoch: 200/1500, loss: 13.666513012586037, correct: 46
Epoch: 210/1500, loss: 13.625416346816985, correct: 46
Epoch: 220/1500, loss: 13.583231357800017, correct: 46
Epoch: 230/1500, loss: 13.539653172529434, correct: 46
Epoch: 240/1500, loss: 13.494614386434868, correct: 46
Epoch: 250/1500, loss: 13.447860686335167, correct: 46
Epoch: 260/1500, loss: 13.398854253505885, correct: 46
Epoch: 270/1500, loss: 13.347232933147508, correct: 46
Epoch: 280/1500, loss: 13.292542206847672, correct: 46
Epoch: 290/1500, loss: 13.234421875869236, correct: 46
Epoch: 300/1500, loss: 13.172646254359668, correct: 46
Epoch: 310/1500, loss: 13.106770846467056, correct: 46
Epoch: 320/1500, loss: 13.034961609701352, correct: 46
Epoch: 330/1500, loss: 12.956190838518484, correct: 46
Epoch: 340/1500, loss: 12.86806724010256, correct: 46
Epoch: 350/1500, loss: 12.770480717494165, correct: 46
Epoch: 360/1500, loss: 12.676939536554887, correct: 46
Epoch: 370/1500, loss: 12.587817854480898, correct: 46
Epoch: 380/1500, loss: 12.49420349026527, correct: 46
Epoch: 390/1500, loss: 12.395823378081968, correct: 46
Epoch: 400/1500, loss: 12.292409321935953, correct: 46
Epoch: 410/1500, loss: 12.183698354300178, correct: 46
Epoch: 420/1500, loss: 12.069433839959137, correct: 46
Epoch: 430/1500, loss: 11.954674076586924, correct: 46
Epoch: 440/1500, loss: 11.839591715701372, correct: 46
Epoch: 450/1500, loss: 11.719720476125925, correct: 46
Epoch: 460/1500, loss: 11.594782485045068, correct: 46
Epoch: 470/1500, loss: 11.46449542734263, correct: 46
Epoch: 480/1500, loss: 11.328566781007499, correct: 46
Epoch: 490/1500, loss: 11.186691403246313, correct: 46
Epoch: 500/1500, loss: 11.038550636008516, correct: 46
Epoch: 510/1500, loss: 10.883812374123222, correct: 46
Epoch: 520/1500, loss: 10.72213207424582, correct: 46
Epoch: 530/1500, loss: 10.55315493428851, correct: 46
Epoch: 540/1500, loss: 10.376519623041393, correct: 46
Epoch: 550/1500, loss: 10.191864057543027, correct: 46
Epoch: 560/1500, loss: 9.998833832504113, correct: 46
Epoch: 570/1500, loss: 9.797093998784426, correct: 46
Epoch: 580/1500, loss: 9.586344946800924, correct: 46
Epoch: 590/1500, loss: 9.366343139345787, correct: 46
Epoch: 600/1500, loss: 9.136927301536478, correct: 46
Epoch: 610/1500, loss: 8.898050339681758, correct: 46
Epoch: 620/1500, loss: 8.649816639698308, correct: 46
Epoch: 630/1500, loss: 8.392523410826906, correct: 46
Epoch: 640/1500, loss: 8.126703360588046, correct: 46
Epoch: 650/1500, loss: 7.8538895560877755, correct: 46
Epoch: 660/1500, loss: 7.575409011098737, correct: 46
Epoch: 670/1500, loss: 7.291940459534759, correct: 46
Epoch: 680/1500, loss: 7.016885441438938, correct: 46
Epoch: 690/1500, loss: 6.811697532329714, correct: 46
Epoch: 700/1500, loss: 6.6156244256870025, correct: 46
Epoch: 710/1500, loss: 6.425989169775264, correct: 46
Epoch: 720/1500, loss: 6.239341086825617, correct: 46
Epoch: 730/1500, loss: 6.054401783789353, correct: 46
Epoch: 740/1500, loss: 5.872259639521713, correct: 46
Epoch: 750/1500, loss: 5.693102135843614, correct: 46
Epoch: 760/1500, loss: 5.5178901494923664, correct: 46
Epoch: 770/1500, loss: 5.364371694452517, correct: 46
Epoch: 780/1500, loss: 5.230843691011318, correct: 46
Epoch: 790/1500, loss: 5.108628032874797, correct: 46
Epoch: 800/1500, loss: 4.993760546380758, correct: 46
Epoch: 810/1500, loss: 4.883081060847663, correct: 46
Epoch: 820/1500, loss: 4.775395223759844, correct: 46
Epoch: 830/1500, loss: 4.670634874137439, correct: 46
Epoch: 840/1500, loss: 4.56879478853938, correct: 46
Epoch: 850/1500, loss: 4.469873130944617, correct: 46
Epoch: 860/1500, loss: 4.373855944323773, correct: 47
Epoch: 870/1500, loss: 4.2850682402586635, correct: 49
Epoch: 880/1500, loss: 4.198453435076298, correct: 49
Epoch: 890/1500, loss: 4.1166022157795155, correct: 49
Epoch: 900/1500, loss: 4.040278994953392, correct: 49
Epoch: 910/1500, loss: 3.9646757655501976, correct: 49
Epoch: 920/1500, loss: 3.8938178364382523, correct: 49
Epoch: 930/1500, loss: 3.8292839543385235, correct: 49
Epoch: 940/1500, loss: 3.763535956175755, correct: 49
Epoch: 950/1500, loss: 3.7046204838804235, correct: 49
Epoch: 960/1500, loss: 3.645819935376982, correct: 49
Epoch: 970/1500, loss: 3.5915446847660775, correct: 49
Epoch: 980/1500, loss: 3.537515323817481, correct: 49
Epoch: 990/1500, loss: 3.483589167754765, correct: 49
Epoch: 1000/1500, loss: 3.4318204226056324, correct: 49
Epoch: 1010/1500, loss: 3.381027765415348, correct: 49
Epoch: 1020/1500, loss: 3.332463035056935, correct: 49
Epoch: 1030/1500, loss: 3.2854735836719913, correct: 49
Epoch: 1040/1500, loss: 3.2401246889778297, correct: 49
Epoch: 1050/1500, loss: 3.1942887743444115, correct: 49
Epoch: 1060/1500, loss: 3.152053250270778, correct: 49
Epoch: 1070/1500, loss: 3.109364615616295, correct: 49
Epoch: 1080/1500, loss: 3.0682861508866535, correct: 49
Epoch: 1090/1500, loss: 3.0284661112802174, correct: 49
Epoch: 1100/1500, loss: 2.989667880445397, correct: 49
Epoch: 1110/1500, loss: 2.951884608455444, correct: 49
Epoch: 1120/1500, loss: 2.9146347876189647, correct: 49
Epoch: 1130/1500, loss: 2.8784848647615098, correct: 49
Epoch: 1140/1500, loss: 2.8431611713711025, correct: 49
Epoch: 1150/1500, loss: 2.8085986081014234, correct: 49
Epoch: 1160/1500, loss: 2.7747671501082958, correct: 49
Epoch: 1170/1500, loss: 2.74164315591398, correct: 49
Epoch: 1180/1500, loss: 2.7092046357562785, correct: 49
Epoch: 1190/1500, loss: 2.677430474564897, correct: 49
Epoch: 1200/1500, loss: 2.6463002951143806, correct: 49
Epoch: 1210/1500, loss: 2.6157944209412674, correct: 49
Epoch: 1220/1500, loss: 2.5858938536950826, correct: 49
Epoch: 1230/1500, loss: 2.5565802523097814, correct: 49
Epoch: 1240/1500, loss: 2.5278359124456906, correct: 49
Epoch: 1250/1500, loss: 2.4996437461532652, correct: 49
Epoch: 1260/1500, loss: 2.4719872618431764, correct: 49
Epoch: 1270/1500, loss: 2.4448505446254343, correct: 49
Epoch: 1280/1500, loss: 2.4182182370577947, correct: 49
Epoch: 1290/1500, loss: 2.392075520330485, correct: 49
Epoch: 1300/1500, loss: 2.366408095906239, correct: 49
Epoch: 1310/1500, loss: 2.3412021676292, correct: 49
Epoch: 1320/1500, loss: 2.316444424312025, correct: 49
Epoch: 1330/1500, loss: 2.2921220228070576, correct: 49
Epoch: 1340/1500, loss: 2.268222571564482, correct: 49
Epoch: 1350/1500, loss: 2.244734114677889, correct: 49
Epoch: 1360/1500, loss: 2.2216451164154245, correct: 49
Epoch: 1370/1500, loss: 2.1989444462329253, correct: 49
Epoch: 1380/1500, loss: 2.1766213642636987, correct: 49
Epoch: 1390/1500, loss: 2.154665507278422, correct: 49
Epoch: 1400/1500, loss: 2.1330668751073176, correct: 49
Epoch: 1410/1500, loss: 2.1118158175159607, correct: 49
Epoch: 1420/1500, loss: 2.0909030215251945, correct: 49
Epoch: 1430/1500, loss: 2.0703194991650635, correct: 49
Epoch: 1440/1500, loss: 2.0500565756521616, correct: 49
Epoch: 1450/1500, loss: 2.0301058779794063, correct: 49
Epoch: 1460/1500, loss: 2.010459323907028, correct: 49
Epoch: 1470/1500, loss: 1.9911091113433264, correct: 49
Epoch: 1480/1500, loss: 1.972047708103686, correct: 49
Epoch: 1490/1500, loss: 1.9532678420363048, correct: 50
Epoch: 1500/1500, loss: 1.934762491503102, correct: 50
```

### Task 2.5.3: Split Dataset
50 data points

**Hyperparameters**:
- learning rate: 0.05
- max epochs: 2000
- hidden layers: 9

<img src="images\task2_5\3. split\1. Dataset.png" width="50%">
<img src="images\task2_5\3. split\2. Hidden Layer and Initial Setting.png" width="50%">
<img src="images\task2_5\3. split\3. Hyperparameters and Results.png" width="50%">
<img src="images\task2_5\3. split\4. Loss Graph and Table.png" width="50%">

**Split Training Log**:
```
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/2000, loss: 0, correct: 0
Epoch: 0/2000, loss: 0, correct: 0
Epoch: 10/2000, loss: 33.985549425615766, correct: 29
Epoch: 20/2000, loss: 33.8225671110268, correct: 29
Epoch: 30/2000, loss: 33.744168812942235, correct: 29
Epoch: 40/2000, loss: 33.67583104383674, correct: 29
Epoch: 50/2000, loss: 33.60766735881086, correct: 29
Epoch: 60/2000, loss: 33.53201320258778, correct: 29
Epoch: 70/2000, loss: 33.44267654897917, correct: 29
Epoch: 80/2000, loss: 33.33239573922046, correct: 29
Epoch: 90/2000, loss: 33.19464720291775, correct: 29
Epoch: 100/2000, loss: 33.032034754188984, correct: 29
Epoch: 110/2000, loss: 32.83927013124746, correct: 30
Epoch: 120/2000, loss: 32.616040778270616, correct: 32
Epoch: 130/2000, loss: 32.3817508801387, correct: 32
Epoch: 140/2000, loss: 32.144144569436385, correct: 32
Epoch: 150/2000, loss: 31.878358943778434, correct: 32
Epoch: 160/2000, loss: 31.602519062409417, correct: 35
Epoch: 170/2000, loss: 31.324065348450326, correct: 37
Epoch: 180/2000, loss: 31.03655351000051, correct: 37
Epoch: 190/2000, loss: 30.726772100011022, correct: 37
Epoch: 200/2000, loss: 30.39506310589611, correct: 40
Epoch: 210/2000, loss: 30.037057680025725, correct: 41
Epoch: 220/2000, loss: 29.67976440962515, correct: 41
Epoch: 230/2000, loss: 29.322850937952428, correct: 41
Epoch: 240/2000, loss: 28.958576401439196, correct: 41
Epoch: 250/2000, loss: 28.58649349180335, correct: 41
Epoch: 260/2000, loss: 28.21977838393075, correct: 41
Epoch: 270/2000, loss: 27.846189653701053, correct: 41
Epoch: 280/2000, loss: 27.448463873573523, correct: 41
Epoch: 290/2000, loss: 27.043724070831637, correct: 41
Epoch: 300/2000, loss: 26.63128784540249, correct: 41
Epoch: 310/2000, loss: 26.227448281018994, correct: 41
Epoch: 320/2000, loss: 25.81842607887442, correct: 41
Epoch: 330/2000, loss: 25.396676446184177, correct: 41
Epoch: 340/2000, loss: 24.980212471149457, correct: 41
Epoch: 350/2000, loss: 24.55431418064442, correct: 41
Epoch: 360/2000, loss: 24.127253490549595, correct: 41
Epoch: 370/2000, loss: 23.690648776197538, correct: 41
Epoch: 380/2000, loss: 23.239565649177774, correct: 41
Epoch: 390/2000, loss: 22.779237708095422, correct: 41
Epoch: 400/2000, loss: 22.312442754744552, correct: 41
Epoch: 410/2000, loss: 21.840752236168072, correct: 41
Epoch: 420/2000, loss: 21.360328883848858, correct: 41
Epoch: 430/2000, loss: 20.868041403167258, correct: 41
Epoch: 440/2000, loss: 20.364452718260743, correct: 41
Epoch: 450/2000, loss: 19.850844279102496, correct: 41
Epoch: 460/2000, loss: 19.340183256749043, correct: 41
Epoch: 470/2000, loss: 18.831848418192344, correct: 41
Epoch: 480/2000, loss: 18.321416010747598, correct: 41
Epoch: 490/2000, loss: 17.813044089021652, correct: 44
Epoch: 500/2000, loss: 17.30769001788318, correct: 44
Epoch: 510/2000, loss: 16.807333839728084, correct: 45
Epoch: 520/2000, loss: 16.312907007337394, correct: 47
Epoch: 530/2000, loss: 15.825047383899642, correct: 47
Epoch: 540/2000, loss: 15.34502424008884, correct: 47
Epoch: 550/2000, loss: 14.87438492145587, correct: 47
Epoch: 560/2000, loss: 14.41521111807435, correct: 47
Epoch: 570/2000, loss: 13.96803593252525, correct: 47
Epoch: 580/2000, loss: 13.533749563632295, correct: 48
Epoch: 590/2000, loss: 13.116457588275022, correct: 48
Epoch: 600/2000, loss: 12.71498877170388, correct: 48
Epoch: 610/2000, loss: 12.328714272694029, correct: 48
Epoch: 620/2000, loss: 11.957317129885283, correct: 49
Epoch: 630/2000, loss: 11.601328594134081, correct: 49
Epoch: 640/2000, loss: 11.259064471543205, correct: 49
Epoch: 650/2000, loss: 10.930418432736282, correct: 49
Epoch: 660/2000, loss: 10.615128212832113, correct: 49
Epoch: 670/2000, loss: 10.312729744839785, correct: 49
Epoch: 680/2000, loss: 10.02284647391366, correct: 49
Epoch: 690/2000, loss: 9.74605225845228, correct: 49
Epoch: 700/2000, loss: 9.48109540747619, correct: 49
Epoch: 710/2000, loss: 9.227023700890108, correct: 49
Epoch: 720/2000, loss: 8.983525809677213, correct: 49
Epoch: 730/2000, loss: 8.74961238852427, correct: 49
Epoch: 740/2000, loss: 8.525874512159286, correct: 49
Epoch: 750/2000, loss: 8.312002771731313, correct: 49
Epoch: 760/2000, loss: 8.10751725909473, correct: 49
Epoch: 770/2000, loss: 7.911646200930607, correct: 49
Epoch: 780/2000, loss: 7.724004931206742, correct: 49
Epoch: 790/2000, loss: 7.544223492273367, correct: 49
Epoch: 800/2000, loss: 7.371862800214667, correct: 49
Epoch: 810/2000, loss: 7.2065572170925005, correct: 49
Epoch: 820/2000, loss: 7.047966617188469, correct: 49
Epoch: 830/2000, loss: 6.895767732220776, correct: 49
Epoch: 840/2000, loss: 6.749653364056861, correct: 49
Epoch: 850/2000, loss: 6.60933030645788, correct: 49
Epoch: 860/2000, loss: 6.4744117184714565, correct: 49
Epoch: 870/2000, loss: 6.344750204989964, correct: 49
Epoch: 880/2000, loss: 6.220225775098139, correct: 49
Epoch: 890/2000, loss: 6.099694702413731, correct: 49
Epoch: 900/2000, loss: 5.983381483151366, correct: 49
Epoch: 910/2000, loss: 5.871486651991342, correct: 49
Epoch: 920/2000, loss: 5.7637961615512925, correct: 49
Epoch: 930/2000, loss: 5.660107685427437, correct: 49
Epoch: 940/2000, loss: 5.560054811495833, correct: 49
Epoch: 950/2000, loss: 5.463660817938139, correct: 49
Epoch: 960/2000, loss: 5.371499687610295, correct: 49
Epoch: 970/2000, loss: 5.28275850576066, correct: 49
Epoch: 980/2000, loss: 5.197129259498345, correct: 49
Epoch: 990/2000, loss: 5.114467081719322, correct: 49
Epoch: 1000/2000, loss: 5.034635090210494, correct: 49
Epoch: 1010/2000, loss: 4.957503853090868, correct: 49
Epoch: 1020/2000, loss: 4.882950915501037, correct: 49
Epoch: 1030/2000, loss: 4.810860363412604, correct: 49
Epoch: 1040/2000, loss: 4.741122420803867, correct: 49
Epoch: 1050/2000, loss: 4.673685385093904, correct: 49
Epoch: 1060/2000, loss: 4.6084061961195575, correct: 49
Epoch: 1070/2000, loss: 4.545177400360965, correct: 49
Epoch: 1080/2000, loss: 4.483962629014061, correct: 49
Epoch: 1090/2000, loss: 4.424721661439533, correct: 49
Epoch: 1100/2000, loss: 4.367262617869796, correct: 49
Epoch: 1110/2000, loss: 4.3115114892491135, correct: 49
Epoch: 1120/2000, loss: 4.257398050049248, correct: 49
Epoch: 1130/2000, loss: 4.204956179120236, correct: 49
Epoch: 1140/2000, loss: 4.15420411578569, correct: 49
Epoch: 1150/2000, loss: 4.104880347397948, correct: 49
Epoch: 1160/2000, loss: 4.056926384153223, correct: 49
Epoch: 1170/2000, loss: 4.010288672390574, correct: 49
Epoch: 1180/2000, loss: 3.9649180674508755, correct: 50
Epoch: 1190/2000, loss: 3.920763895193825, correct: 50
Epoch: 1200/2000, loss: 3.877780375471272, correct: 50
Epoch: 1210/2000, loss: 3.8359237115632054, correct: 50
Epoch: 1220/2000, loss: 3.7951556149046297, correct: 50
Epoch: 1230/2000, loss: 3.7554644461020303, correct: 50
Epoch: 1240/2000, loss: 3.716783487346797, correct: 50
Epoch: 1250/2000, loss: 3.679080243785939, correct: 50
Epoch: 1260/2000, loss: 3.6423074288269333, correct: 50
Epoch: 1270/2000, loss: 3.6064285775241385, correct: 50
Epoch: 1280/2000, loss: 3.5714117851201466, correct: 50
Epoch: 1290/2000, loss: 3.537226548742454, correct: 50
Epoch: 1300/2000, loss: 3.5036320029289185, correct: 50
Epoch: 1310/2000, loss: 3.470758980327102, correct: 50
Epoch: 1320/2000, loss: 3.4386393033585176, correct: 50
Epoch: 1330/2000, loss: 3.407259506161133, correct: 50
Epoch: 1340/2000, loss: 3.3765938110961016, correct: 50
Epoch: 1350/2000, loss: 3.3466226970640336, correct: 50
Epoch: 1360/2000, loss: 3.3173198640796144, correct: 50
Epoch: 1370/2000, loss: 3.2886099025141764, correct: 50
Epoch: 1380/2000, loss: 3.2603465848063435, correct: 50
Epoch: 1390/2000, loss: 3.232705818611818, correct: 50
Epoch: 1400/2000, loss: 3.205667667846095, correct: 50
Epoch: 1410/2000, loss: 3.1792127531944487, correct: 50
Epoch: 1420/2000, loss: 3.1532234478794496, correct: 50
Epoch: 1430/2000, loss: 3.1276046388171928, correct: 50
Epoch: 1440/2000, loss: 3.102220110056887, correct: 50
Epoch: 1450/2000, loss: 3.0771535334680817, correct: 50
Epoch: 1460/2000, loss: 3.0526383450442394, correct: 50
Epoch: 1470/2000, loss: 3.0286537978700583, correct: 50
Epoch: 1480/2000, loss: 3.005181663026119, correct: 50
Epoch: 1490/2000, loss: 2.9767330874698565, correct: 50
Epoch: 1500/2000, loss: 2.9468504230234536, correct: 50
Epoch: 1510/2000, loss: 2.917848155201549, correct: 50
Epoch: 1520/2000, loss: 2.893841597967273, correct: 50
Epoch: 1530/2000, loss: 2.872212805698288, correct: 50
Epoch: 1540/2000, loss: 2.8510036376580588, correct: 50
Epoch: 1550/2000, loss: 2.83023244830429, correct: 50
Epoch: 1560/2000, loss: 2.809882790155406, correct: 50
Epoch: 1570/2000, loss: 2.789892344710849, correct: 50
Epoch: 1580/2000, loss: 2.77023873604302, correct: 50
Epoch: 1590/2000, loss: 2.750989005302193, correct: 50
Epoch: 1600/2000, loss: 2.732101191186305, correct: 50
Epoch: 1610/2000, loss: 2.713255293432789, correct: 50
Epoch: 1620/2000, loss: 2.695019092447256, correct: 50
Epoch: 1630/2000, loss: 2.677130191308194, correct: 50
Epoch: 1640/2000, loss: 2.659665100287291, correct: 50
Epoch: 1650/2000, loss: 2.6425217125241245, correct: 50
Epoch: 1660/2000, loss: 2.6256664044028706, correct: 50
Epoch: 1670/2000, loss: 2.6090907045814657, correct: 50
Epoch: 1680/2000, loss: 2.5929000744909088, correct: 50
Epoch: 1690/2000, loss: 2.577047085217009, correct: 50
Epoch: 1700/2000, loss: 2.5614485780753307, correct: 50
Epoch: 1710/2000, loss: 2.5460995998330893, correct: 50
Epoch: 1720/2000, loss: 2.530981787771273, correct: 50
Epoch: 1730/2000, loss: 2.516083343325663, correct: 50
Epoch: 1740/2000, loss: 2.501404988062849, correct: 50
Epoch: 1750/2000, loss: 2.4869804937854414, correct: 50
Epoch: 1760/2000, loss: 2.472772694942896, correct: 50
Epoch: 1770/2000, loss: 2.458794426581804, correct: 50
Epoch: 1780/2000, loss: 2.4449921653626734, correct: 50
Epoch: 1790/2000, loss: 2.4314472823688544, correct: 50
Epoch: 1800/2000, loss: 2.418633105900621, correct: 50
Epoch: 1810/2000, loss: 2.4058814372074204, correct: 50
Epoch: 1820/2000, loss: 2.3933975050696956, correct: 50
Epoch: 1830/2000, loss: 2.380948269696879, correct: 50
Epoch: 1840/2000, loss: 2.368763463340316, correct: 50
Epoch: 1850/2000, loss: 2.356603875026991, correct: 50
Epoch: 1860/2000, loss: 2.344699859826037, correct: 50
Epoch: 1870/2000, loss: 2.33285297848574, correct: 50
Epoch: 1880/2000, loss: 2.321246418453314, correct: 50
Epoch: 1890/2000, loss: 2.3096767471551267, correct: 50
Epoch: 1900/2000, loss: 2.2982417755183606, correct: 50
Epoch: 1910/2000, loss: 2.286996495415094, correct: 50
Epoch: 1920/2000, loss: 2.2758417329252145, correct: 50
Epoch: 1930/2000, loss: 2.26480391031053, correct: 50
Epoch: 1940/2000, loss: 2.2539151452169617, correct: 50
Epoch: 1950/2000, loss: 2.243151106179168, correct: 50
Epoch: 1960/2000, loss: 2.232481627882287, correct: 50
Epoch: 1970/2000, loss: 2.221890310046052, correct: 50
Epoch: 1980/2000, loss: 2.211452906758705, correct: 50
Epoch: 1990/2000, loss: 2.201128372212253, correct: 50
Epoch: 2000/2000, loss: 2.1909402344804967, correct: 50
```

### Task 2.5.4: XOR Dataset
50 data points

**Hyperparameters**:
- learning rate: 0.05
- max epochs: 2000
- hidden layers: 15

<img src="images\task2_5\4. xor\1. Dataset.png" width="50%">
<img src="images\task2_5\4. xor\2. Hidden Layer and Initial Setting.png" width="50%">
<img src="images\task2_5\4. xor\3. Hyperparameters and Results.png" width="50%">
<img src="images\task2_5\4. xor\4. Loss Graph and Table.png" width="50%">

**XOR Training Log**:
```
Epoch: 0/2000, loss: 0, correct: 0
Epoch: 0/2000, loss: 0, correct: 0
Epoch: 10/2000, loss: 33.50191556567808, correct: 30
Epoch: 20/2000, loss: 31.917690738075212, correct: 35
Epoch: 30/2000, loss: 31.153598637516666, correct: 33
Epoch: 40/2000, loss: 30.45212875375951, correct: 34
Epoch: 50/2000, loss: 29.485464513104002, correct: 38
Epoch: 60/2000, loss: 28.844179626760507, correct: 39
Epoch: 70/2000, loss: 28.332625506022538, correct: 40
Epoch: 80/2000, loss: 27.85260106083724, correct: 39
Epoch: 90/2000, loss: 27.394292164706343, correct: 41
Epoch: 100/2000, loss: 26.92050786286442, correct: 41
Epoch: 110/2000, loss: 26.414533542921617, correct: 41
Epoch: 120/2000, loss: 25.917514494581877, correct: 42
Epoch: 130/2000, loss: 25.42381364258596, correct: 43
Epoch: 140/2000, loss: 24.931374312482756, correct: 43
Epoch: 150/2000, loss: 24.439060706045357, correct: 43
Epoch: 160/2000, loss: 23.94001051629886, correct: 43
Epoch: 170/2000, loss: 23.439511348880146, correct: 43
Epoch: 180/2000, loss: 22.935687099777795, correct: 44
Epoch: 190/2000, loss: 22.351859552864504, correct: 45
Epoch: 200/2000, loss: 21.55752816150684, correct: 46
Epoch: 210/2000, loss: 21.048703636751018, correct: 45
Epoch: 220/2000, loss: 20.567892169269264, correct: 46
Epoch: 230/2000, loss: 20.092864303070915, correct: 46
Epoch: 240/2000, loss: 19.62429209965463, correct: 46
Epoch: 250/2000, loss: 19.162054240547988, correct: 46
Epoch: 260/2000, loss: 18.70663240967942, correct: 47
Epoch: 270/2000, loss: 18.258613398234736, correct: 47
Epoch: 280/2000, loss: 17.819279566484877, correct: 47
Epoch: 290/2000, loss: 17.38834927693774, correct: 47
Epoch: 300/2000, loss: 16.96642522504241, correct: 48
Epoch: 310/2000, loss: 16.555949455913183, correct: 48
Epoch: 320/2000, loss: 16.15539286214183, correct: 48
Epoch: 330/2000, loss: 15.76510005174547, correct: 48
Epoch: 340/2000, loss: 15.384558964711857, correct: 48
Epoch: 350/2000, loss: 15.014635803826994, correct: 48
Epoch: 360/2000, loss: 14.654732554571781, correct: 48
Epoch: 370/2000, loss: 14.304803306696684, correct: 48
Epoch: 380/2000, loss: 13.965218404746336, correct: 50
Epoch: 390/2000, loss: 13.635507511199439, correct: 50
Epoch: 400/2000, loss: 13.314948795883177, correct: 50
Epoch: 410/2000, loss: 13.00271637585714, correct: 49
Epoch: 420/2000, loss: 12.700402670360289, correct: 49
Epoch: 430/2000, loss: 12.407360390423719, correct: 49
Epoch: 440/2000, loss: 12.122962970869983, correct: 49
Epoch: 450/2000, loss: 11.847670172579525, correct: 49
Epoch: 460/2000, loss: 11.580713066315544, correct: 49
Epoch: 470/2000, loss: 11.321777487702086, correct: 49
Epoch: 480/2000, loss: 11.07107311800956, correct: 49
Epoch: 490/2000, loss: 10.828779758520579, correct: 49
Epoch: 500/2000, loss: 10.593967498196822, correct: 49
Epoch: 510/2000, loss: 10.362244388710993, correct: 49
Epoch: 520/2000, loss: 10.137793976805199, correct: 49
Epoch: 530/2000, loss: 9.920853668556555, correct: 49
Epoch: 540/2000, loss: 9.711199861775398, correct: 49
Epoch: 550/2000, loss: 9.503913658403906, correct: 49
Epoch: 560/2000, loss: 9.297316660338376, correct: 49
Epoch: 570/2000, loss: 9.100189290535994, correct: 49
Epoch: 580/2000, loss: 8.913265334784118, correct: 49
Epoch: 590/2000, loss: 8.736320429091922, correct: 49
Epoch: 600/2000, loss: 8.566788865790622, correct: 49
Epoch: 610/2000, loss: 8.402270436499597, correct: 49
Epoch: 620/2000, loss: 8.240183685743805, correct: 49
Epoch: 630/2000, loss: 8.082977964930295, correct: 49
Epoch: 640/2000, loss: 7.932984641188439, correct: 49
Epoch: 650/2000, loss: 7.787466923392979, correct: 49
Epoch: 660/2000, loss: 7.646948451187367, correct: 49
Epoch: 670/2000, loss: 7.510514575341373, correct: 49
Epoch: 680/2000, loss: 7.376900983348227, correct: 49
Epoch: 690/2000, loss: 7.2494319454903415, correct: 49
Epoch: 700/2000, loss: 7.123906022152716, correct: 49
Epoch: 710/2000, loss: 7.003367951317429, correct: 49
Epoch: 720/2000, loss: 6.889838879018193, correct: 49
Epoch: 730/2000, loss: 6.7762786192768205, correct: 49
Epoch: 740/2000, loss: 6.666827517292296, correct: 49
Epoch: 750/2000, loss: 6.562525801331792, correct: 49
Epoch: 760/2000, loss: 6.462082925770912, correct: 49
Epoch: 770/2000, loss: 6.364366417237202, correct: 49
Epoch: 780/2000, loss: 6.265786237341317, correct: 49
Epoch: 790/2000, loss: 6.175170491462282, correct: 49
Epoch: 800/2000, loss: 6.086110380344686, correct: 49
Epoch: 810/2000, loss: 5.9974398553034725, correct: 49
Epoch: 820/2000, loss: 5.913147468917001, correct: 49
Epoch: 830/2000, loss: 5.8287980847606375, correct: 49
Epoch: 840/2000, loss: 5.749171626045924, correct: 49
Epoch: 850/2000, loss: 5.671854925692592, correct: 49
Epoch: 860/2000, loss: 5.597346870494282, correct: 49
Epoch: 870/2000, loss: 5.5224689668051195, correct: 49
Epoch: 880/2000, loss: 5.450166777359568, correct: 49
Epoch: 890/2000, loss: 5.380155970420505, correct: 49
Epoch: 900/2000, loss: 5.3126701020730325, correct: 49
Epoch: 910/2000, loss: 5.247873272406926, correct: 49
Epoch: 920/2000, loss: 5.182179097172673, correct: 49
Epoch: 930/2000, loss: 5.118416915830229, correct: 49
Epoch: 940/2000, loss: 5.057776882471425, correct: 49
Epoch: 950/2000, loss: 4.998945955027719, correct: 49
Epoch: 960/2000, loss: 4.9386476201132545, correct: 49
Epoch: 970/2000, loss: 4.8819996645616905, correct: 49
Epoch: 980/2000, loss: 4.826850035208354, correct: 49
Epoch: 990/2000, loss: 4.772623527103201, correct: 49
Epoch: 1000/2000, loss: 4.718720180169598, correct: 49
Epoch: 1010/2000, loss: 4.664798415735465, correct: 49
Epoch: 1020/2000, loss: 4.612548313167632, correct: 49
Epoch: 1030/2000, loss: 4.562340832152713, correct: 49
Epoch: 1040/2000, loss: 4.5128157874027455, correct: 49
Epoch: 1050/2000, loss: 4.464350310989137, correct: 49
Epoch: 1060/2000, loss: 4.416698808888642, correct: 49
Epoch: 1070/2000, loss: 4.370395116511364, correct: 49
Epoch: 1080/2000, loss: 4.323995955480835, correct: 49
Epoch: 1090/2000, loss: 4.278870582011117, correct: 50
Epoch: 1100/2000, loss: 4.234544924818499, correct: 50
Epoch: 1110/2000, loss: 4.190882630369557, correct: 50
Epoch: 1120/2000, loss: 4.148563887219057, correct: 50
Epoch: 1130/2000, loss: 4.106623996913396, correct: 50
Epoch: 1140/2000, loss: 4.065579217614989, correct: 50
Epoch: 1150/2000, loss: 4.02523249844311, correct: 50
Epoch: 1160/2000, loss: 3.9855947275996537, correct: 50
Epoch: 1170/2000, loss: 3.94670181852691, correct: 50
Epoch: 1180/2000, loss: 3.9086978727673802, correct: 50
Epoch: 1190/2000, loss: 3.8712861547153326, correct: 50
Epoch: 1200/2000, loss: 3.8346201369767736, correct: 50
Epoch: 1210/2000, loss: 3.798681923539302, correct: 50
Epoch: 1220/2000, loss: 3.7632934874537356, correct: 50
Epoch: 1230/2000, loss: 3.7283176080833353, correct: 50
Epoch: 1240/2000, loss: 3.694066445503157, correct: 50
Epoch: 1250/2000, loss: 3.660594516353499, correct: 50
Epoch: 1260/2000, loss: 3.6273924813178393, correct: 50
Epoch: 1270/2000, loss: 3.5947715529574245, correct: 50
Epoch: 1280/2000, loss: 3.5621095327109877, correct: 50
Epoch: 1290/2000, loss: 3.5300989093445683, correct: 50
Epoch: 1300/2000, loss: 3.498370797915292, correct: 50
Epoch: 1310/2000, loss: 3.4672590830287198, correct: 50
Epoch: 1320/2000, loss: 3.436444166341428, correct: 50
Epoch: 1330/2000, loss: 3.4069639133311247, correct: 50
Epoch: 1340/2000, loss: 3.378000567490772, correct: 50
Epoch: 1350/2000, loss: 3.3494090276208457, correct: 50
Epoch: 1360/2000, loss: 3.321257850517823, correct: 50
Epoch: 1370/2000, loss: 3.293537400735211, correct: 50
Epoch: 1380/2000, loss: 3.2662584211858254, correct: 50
Epoch: 1390/2000, loss: 3.2395055892942004, correct: 50
Epoch: 1400/2000, loss: 3.2129185038568906, correct: 50
Epoch: 1410/2000, loss: 3.1866873047338595, correct: 50
Epoch: 1420/2000, loss: 3.1600325282715005, correct: 50
Epoch: 1430/2000, loss: 3.133721177711182, correct: 50
Epoch: 1440/2000, loss: 3.1078601010584532, correct: 50
Epoch: 1450/2000, loss: 3.082380595869378, correct: 50
Epoch: 1460/2000, loss: 3.0581600980652266, correct: 50
Epoch: 1470/2000, loss: 3.034287846524739, correct: 50
Epoch: 1480/2000, loss: 3.0107508727577925, correct: 50
Epoch: 1490/2000, loss: 2.9875260460496076, correct: 50
Epoch: 1500/2000, loss: 2.964766391664095, correct: 50
Epoch: 1510/2000, loss: 2.9421627362439877, correct: 50
Epoch: 1520/2000, loss: 2.91988392018469, correct: 50
Epoch: 1530/2000, loss: 2.8966635076488614, correct: 50
Epoch: 1540/2000, loss: 2.8737233573933816, correct: 50
Epoch: 1550/2000, loss: 2.8515768930084517, correct: 50
Epoch: 1560/2000, loss: 2.8305048593542237, correct: 50
Epoch: 1570/2000, loss: 2.8099117400535416, correct: 50
Epoch: 1580/2000, loss: 2.7895604253941118, correct: 50
Epoch: 1590/2000, loss: 2.7694137305974915, correct: 50
Epoch: 1600/2000, loss: 2.7488600880064116, correct: 50
Epoch: 1610/2000, loss: 2.7275398949069043, correct: 50
Epoch: 1620/2000, loss: 2.707261725476811, correct: 50
Epoch: 1630/2000, loss: 2.6881933780382576, correct: 50
Epoch: 1640/2000, loss: 2.6693091876825155, correct: 50
Epoch: 1650/2000, loss: 2.6508179363886315, correct: 50
Epoch: 1660/2000, loss: 2.6322761232188734, correct: 50
Epoch: 1670/2000, loss: 2.61204127695981, correct: 50
Epoch: 1680/2000, loss: 2.5933964252457957, correct: 50
Epoch: 1690/2000, loss: 2.5758418259354032, correct: 50
Epoch: 1700/2000, loss: 2.55858209466308, correct: 50
Epoch: 1710/2000, loss: 2.5414054024275763, correct: 50
Epoch: 1720/2000, loss: 2.522360215563343, correct: 50
Epoch: 1730/2000, loss: 2.5046725694858507, correct: 50
Epoch: 1740/2000, loss: 2.488100383626517, correct: 50
Epoch: 1750/2000, loss: 2.471959831065819, correct: 50
Epoch: 1760/2000, loss: 2.45552954206544, correct: 50
Epoch: 1770/2000, loss: 2.4372368305364067, correct: 50
Epoch: 1780/2000, loss: 2.421315507620983, correct: 50
Epoch: 1790/2000, loss: 2.405815699236556, correct: 50
Epoch: 1800/2000, loss: 2.390647294064526, correct: 50
Epoch: 1810/2000, loss: 2.3733166166926734, correct: 50
Epoch: 1820/2000, loss: 2.3578047815066006, correct: 50
Epoch: 1830/2000, loss: 2.3430741790670555, correct: 50
Epoch: 1840/2000, loss: 2.328613617184378, correct: 50
Epoch: 1850/2000, loss: 2.3115960334882404, correct: 50
Epoch: 1860/2000, loss: 2.2971684193234942, correct: 50
Epoch: 1870/2000, loss: 2.283060397911858, correct: 50
Epoch: 1880/2000, loss: 2.2685174731573268, correct: 50
Epoch: 1890/2000, loss: 2.254711384590244, correct: 50
Epoch: 1900/2000, loss: 2.2410196100156803, correct: 50
Epoch: 1910/2000, loss: 2.226175276449828, correct: 50
Epoch: 1920/2000, loss: 2.210716204496177, correct: 50
Epoch: 1930/2000, loss: 2.1982635710222485, correct: 50
Epoch: 1940/2000, loss: 2.1841336026972256, correct: 50
Epoch: 1950/2000, loss: 2.1752427092870157, correct: 50
Epoch: 1960/2000, loss: 2.157968871781341, correct: 50
Epoch: 1970/2000, loss: 2.150486421971223, correct: 50
Epoch: 1980/2000, loss: 2.132587636846978, correct: 50
Epoch: 1990/2000, loss: 2.120848771933832, correct: 50
Epoch: 2000/2000, loss: 2.1069766887539307, correct: 50
```