import re
import matplotlib.pyplot as plt
import sys
print("ab")
# Provided text (including additional parameters)
text3 = """
/home/yoavharlap/miniconda3/envs/cryo_classification/bin/python /home/yoavharlap/PycharmProjects/cryo_classification/train.py
MNIST: False
Cuda Available!!!!!!!!
new graph starts here
the lr is: 1e-05
the batch size is: 64
the number of epochs is: 150
Saving Best Model with Accuracy:  68.73807525634766
Epoch: 1 / 150 Accuracy : 68.73807525634766 % loss: 1.0907925367355347 test_acc: 67.60333251953125
Epoch: 2 / 150 Accuracy : 54.538021087646484 % loss: 0.8572503924369812 test_acc: 53.68659973144531
Saving Best Model with Accuracy:  72.8536376953125
Epoch: 3 / 150 Accuracy : 72.8536376953125 % loss: 1.025017499923706 test_acc: 71.36412048339844
Epoch: 4 / 150 Accuracy : 72.41754913330078 % loss: 0.9540311098098755 test_acc: 70.14988708496094
Saving Best Model with Accuracy:  73.28972625732422
Epoch: 5 / 150 Accuracy : 73.28972625732422 % loss: 0.8556891083717346 test_acc: 70.40726470947266
Saving Best Model with Accuracy:  74.1346435546875
Epoch: 6 / 150 Accuracy : 74.1346435546875 % loss: 0.798534095287323 test_acc: 71.26721954345703
Epoch: 7 / 150 Accuracy : 73.34423828125 % loss: 0.38490232825279236 test_acc: 71.79712677001953
Epoch: 8 / 150 Accuracy : 73.39874267578125 % loss: 0.3252089023590088 test_acc: 71.79409790039062
Epoch: 9 / 150 Accuracy : 73.39874267578125 % loss: 0.2886747717857361 test_acc: 71.79409790039062
Epoch: 10 / 150 Accuracy : 73.97110748291016 % loss: 0.3048977255821228 test_acc: 71.79106903076172
Saving Best Model with Accuracy:  75.7154541015625
Epoch: 11 / 150 Accuracy : 75.7154541015625 % loss: 0.28366443514823914 test_acc: 71.72142028808594
Saving Best Model with Accuracy:  76.17879486083984
Epoch: 12 / 150 Accuracy : 76.17879486083984 % loss: 0.21698911488056183 test_acc: 71.72747802734375
Saving Best Model with Accuracy:  76.80567169189453
Epoch: 13 / 150 Accuracy : 76.80567169189453 % loss: 0.19760647416114807 test_acc: 71.68811798095703
Epoch: 14 / 150 Accuracy : 76.53311157226562 % loss: 0.17992763221263885 test_acc: 71.73050689697266
Epoch: 15 / 150 Accuracy : 75.25211334228516 % loss: 0.17200668156147003 test_acc: 71.80923461914062
Epoch: 16 / 150 Accuracy : 75.1430892944336 % loss: 0.2012327015399933 test_acc: 71.79106903076172
Saving Best Model with Accuracy:  84.62796783447266
Epoch: 17 / 150 Accuracy : 84.62796783447266 % loss: 0.18969251215457916 test_acc: 64.71460723876953
Saving Best Model with Accuracy:  85.3093490600586
Epoch: 18 / 150 Accuracy : 85.3093490600586 % loss: 0.15140485763549805 test_acc: 65.11430358886719
Saving Best Model with Accuracy:  86.29054260253906
Epoch: 19 / 150 Accuracy : 86.29054260253906 % loss: 0.14830441772937775 test_acc: 69.28690338134766
Saving Best Model with Accuracy:  86.50858306884766
Epoch: 20 / 150 Accuracy : 86.50858306884766 % loss: 0.13756993412971497 test_acc: 69.6654052734375
Epoch: 21 / 150 Accuracy : 78.76805877685547 % loss: 0.1876647174358368 test_acc: 71.72747802734375
Epoch: 22 / 150 Accuracy : 85.28209686279297 % loss: 0.12835465371608734 test_acc: 70.81604766845703
Epoch: 23 / 150 Accuracy : 74.925048828125 % loss: 0.2500555217266083 test_acc: 71.77289581298828
Epoch: 24 / 150 Accuracy : 73.75306701660156 % loss: 0.13634414970874786 test_acc: 71.81529235839844
Epoch: 25 / 150 Accuracy : 74.37994384765625 % loss: 0.10791915655136108 test_acc: 71.82437896728516
Epoch: 26 / 150 Accuracy : 74.57072448730469 % loss: 0.10220921784639359 test_acc: 71.80923461914062
Epoch: 27 / 150 Accuracy : 75.7154541015625 % loss: 0.09393472969532013 test_acc: 71.82135009765625
Epoch: 28 / 150 Accuracy : 78.27745819091797 % loss: 0.09229685366153717 test_acc: 71.82135009765625
Epoch: 29 / 150 Accuracy : 84.32815551757812 % loss: 0.09426015615463257 test_acc: 71.42467498779297
Saving Best Model with Accuracy:  88.4709701538086
Epoch: 30 / 150 Accuracy : 88.4709701538086 % loss: 0.08707736432552338 test_acc: 70.49507904052734
Epoch: 31 / 150 Accuracy : 85.9634780883789 % loss: 0.0961252823472023 test_acc: 71.27024841308594
Epoch: 32 / 150 Accuracy : 85.33660125732422 % loss: 0.09228057414293289 test_acc: 71.35200500488281
Epoch: 33 / 150 Accuracy : 73.45325469970703 % loss: 0.3400065004825592 test_acc: 71.79409790039062
Epoch: 34 / 150 Accuracy : 77.02371215820312 % loss: 0.8951131701469421 test_acc: 71.81529235839844
Epoch: 35 / 150 Accuracy : 73.23521423339844 % loss: 0.2541169822216034 test_acc: 71.77592468261719
Epoch: 36 / 150 Accuracy : 74.2164077758789 % loss: 0.2138739973306656 test_acc: 71.79409790039062
Epoch: 37 / 150 Accuracy : 75.98800659179688 % loss: 0.2650018036365509 test_acc: 71.83648681640625
Epoch: 38 / 150 Accuracy : 77.62332916259766 % loss: 0.23420923948287964 test_acc: 71.81226348876953
Epoch: 39 / 150 Accuracy : 85.3093490600586 % loss: 0.27862346172332764 test_acc: 71.4398193359375
Epoch: 40 / 150 Accuracy : 87.65331268310547 % loss: 0.2562401294708252 test_acc: 71.28842163085938
Saving Best Model with Accuracy:  88.85254669189453
Epoch: 41 / 150 Accuracy : 88.85254669189453 % loss: 0.22487570345401764 test_acc: 70.99772644042969
Epoch: 42 / 150 Accuracy : 75.7154541015625 % loss: 0.0815725177526474 test_acc: 71.80923461914062
Epoch: 43 / 150 Accuracy : 87.92586517333984 % loss: 0.23225264251232147 test_acc: 71.25814056396484
Saving Best Model with Accuracy:  91.93240356445312
Epoch: 44 / 150 Accuracy : 91.93240356445312 % loss: 0.2613772749900818 test_acc: 69.03255462646484
Saving Best Model with Accuracy:  92.9135971069336
Epoch: 45 / 150 Accuracy : 92.9135971069336 % loss: 0.22072817385196686 test_acc: 66.66767883300781
Epoch: 46 / 150 Accuracy : 91.9051513671875 % loss: 0.19436264038085938 test_acc: 63.39742660522461
Epoch: 47 / 150 Accuracy : 92.31398010253906 % loss: 0.12004587799310684 test_acc: 64.03028106689453
Saving Best Model with Accuracy:  92.96810913085938
Epoch: 48 / 150 Accuracy : 92.96810913085938 % loss: 0.13725753128528595 test_acc: 64.33610534667969
Epoch: 49 / 150 Accuracy : 92.1231918334961 % loss: 0.10079134255647659 test_acc: 63.1763801574707
Saving Best Model with Accuracy:  93.54047393798828
Epoch: 50 / 150 Accuracy : 93.54047393798828 % loss: 0.11778421700000763 test_acc: 65.00227355957031
Epoch: 51 / 150 Accuracy : 89.23412322998047 % loss: 0.10605601966381073 test_acc: 60.003028869628906
Epoch: 52 / 150 Accuracy : 91.93240356445312 % loss: 0.041995856910943985 test_acc: 70.32853698730469
Epoch: 53 / 150 Accuracy : 91.46906280517578 % loss: 0.08877910673618317 test_acc: 61.813777923583984
Saving Best Model with Accuracy:  94.49441528320312
Epoch: 54 / 150 Accuracy : 94.49441528320312 % loss: 0.04962048679590225 test_acc: 65.65934753417969
Epoch: 55 / 150 Accuracy : 94.24911499023438 % loss: 0.09437938034534454 test_acc: 65.51400756835938
Epoch: 56 / 150 Accuracy : 94.38539123535156 % loss: 0.04739980027079582 test_acc: 64.83876037597656
Epoch: 57 / 150 Accuracy : 81.76615142822266 % loss: 0.05299340933561325 test_acc: 71.73656463623047
Epoch: 58 / 150 Accuracy : 81.6571273803711 % loss: 0.048705004155635834 test_acc: 71.74564361572266
Epoch: 59 / 150 Accuracy : 80.97574615478516 % loss: 0.04642447456717491 test_acc: 71.7698745727539
Epoch: 60 / 150 Accuracy : 81.0302505493164 % loss: 0.04432868957519531 test_acc: 71.77289581298828
Epoch: 61 / 150 Accuracy : 81.95693969726562 % loss: 0.044724784791469574 test_acc: 71.73050689697266
Epoch: 62 / 150 Accuracy : 85.36386108398438 % loss: 0.05390328913927078 test_acc: 71.63058471679688
Epoch: 63 / 150 Accuracy : 94.00381469726562 % loss: 0.1129109188914299 test_acc: 69.5594253540039
Epoch: 64 / 150 Accuracy : 90.2425765991211 % loss: 0.24658381938934326 test_acc: 59.491294860839844
Epoch: 65 / 150 Accuracy : 71.21831512451172 % loss: 0.2592274844646454 test_acc: 47.77289962768555
Epoch: 66 / 150 Accuracy : 65.95802307128906 % loss: 0.22381408512592316 test_acc: 45.48069763183594
Epoch: 67 / 150 Accuracy : 59.33496856689453 % loss: 0.19537393748760223 test_acc: 42.66162109375
Epoch: 68 / 150 Accuracy : 56.93649673461914 % loss: 0.1668345183134079 test_acc: 41.559425354003906
Epoch: 69 / 150 Accuracy : 56.00981140136719 % loss: 0.14509500563144684 test_acc: 40.99015808105469
Epoch: 70 / 150 Accuracy : 55.546470642089844 % loss: 0.12835952639579773 test_acc: 40.78728103637695
Epoch: 71 / 150 Accuracy : 55.46470260620117 % loss: 0.11523996293544769 test_acc: 40.672218322753906
Epoch: 72 / 150 Accuracy : 55.573726654052734 % loss: 0.10421163588762283 test_acc: 40.74186325073242
Epoch: 73 / 150 Accuracy : 55.71000289916992 % loss: 0.09447506815195084 test_acc: 40.914459228515625
Epoch: 74 / 150 Accuracy : 55.9825553894043 % loss: 0.08575374633073807 test_acc: 41.080997467041016
Epoch: 75 / 150 Accuracy : 56.445899963378906 % loss: 0.07835261523723602 test_acc: 41.244510650634766
Epoch: 76 / 150 Accuracy : 57.10002899169922 % loss: 0.0719079077243805 test_acc: 41.5109748840332
Epoch: 77 / 150 Accuracy : 57.64513397216797 % loss: 0.0663929134607315 test_acc: 41.79560852050781
Epoch: 78 / 150 Accuracy : 58.43553924560547 % loss: 0.06131356209516525 test_acc: 42.022708892822266
Epoch: 79 / 150 Accuracy : 59.33496856689453 % loss: 0.05688856169581413 test_acc: 42.307342529296875
Epoch: 80 / 150 Accuracy : 60.070865631103516 % loss: 0.052965544164180756 test_acc: 42.59500503540039
Epoch: 81 / 150 Accuracy : 60.67048263549805 % loss: 0.04926890507340431 test_acc: 42.88266372680664
Epoch: 82 / 150 Accuracy : 61.76069641113281 % loss: 0.04570871964097023 test_acc: 43.31869888305664
Epoch: 83 / 150 Accuracy : 62.878170013427734 % loss: 0.04231346398591995 test_acc: 43.86071014404297
Epoch: 84 / 150 Accuracy : 65.05860137939453 % loss: 0.03889802098274231 test_acc: 44.738834381103516
Epoch: 85 / 150 Accuracy : 68.4110107421875 % loss: 0.035824112594127655 test_acc: 46.089324951171875
Epoch: 86 / 150 Accuracy : 75.33387756347656 % loss: 0.03405456617474556 test_acc: 48.96290588378906
Epoch: 87 / 150 Accuracy : 87.84410095214844 % loss: 0.03986151143908501 test_acc: 55.524600982666016
Epoch: 88 / 150 Accuracy : 85.39111328125 % loss: 0.0383605882525444 test_acc: 53.913700103759766
Saving Best Model with Accuracy:  96.15699005126953
Epoch: 89 / 150 Accuracy : 96.15699005126953 % loss: 0.058151744306087494 test_acc: 69.37168884277344
Saving Best Model with Accuracy:  97.16544342041016
Epoch: 90 / 150 Accuracy : 97.16544342041016 % loss: 0.038253407925367355 test_acc: 68.17864990234375
Saving Best Model with Accuracy:  97.7105484008789
Epoch: 91 / 150 Accuracy : 97.7105484008789 % loss: 0.03310501575469971 test_acc: 66.79788208007812
Saving Best Model with Accuracy:  97.90133666992188
Epoch: 92 / 150 Accuracy : 97.90133666992188 % loss: 0.03081675060093403 test_acc: 65.20817565917969
Epoch: 93 / 150 Accuracy : 97.43799591064453 % loss: 0.030158037319779396 test_acc: 63.07040023803711
Epoch: 94 / 150 Accuracy : 94.24911499023438 % loss: 0.025621473789215088 test_acc: 59.37925720214844
Epoch: 95 / 150 Accuracy : 89.37039947509766 % loss: 0.025020264089107513 test_acc: 56.29371643066406
Epoch: 96 / 150 Accuracy : 96.2660140991211 % loss: 0.022153308615088463 test_acc: 61.55033874511719
Saving Best Model with Accuracy:  98.58271789550781
Epoch: 97 / 150 Accuracy : 98.58271789550781 % loss: 0.02299380674958229 test_acc: 65.97728729248047
Epoch: 98 / 150 Accuracy : 98.01036071777344 % loss: 0.023031949996948242 test_acc: 67.59727478027344
Epoch: 99 / 150 Accuracy : 97.51976013183594 % loss: 0.02224903181195259 test_acc: 68.36638641357422
Saving Best Model with Accuracy:  98.82801818847656
Epoch: 100 / 150 Accuracy : 98.82801818847656 % loss: 0.025880703702569008 test_acc: 66.48599243164062
Epoch: 101 / 150 Accuracy : 82.12046813964844 % loss: 0.03302676975727081 test_acc: 51.400455474853516
Epoch: 102 / 150 Accuracy : 84.08285522460938 % loss: 0.03753859922289848 test_acc: 52.44208908081055
Epoch: 103 / 150 Accuracy : 79.91278076171875 % loss: 0.03351214900612831 test_acc: 50.23164367675781
Epoch: 104 / 150 Accuracy : 79.80376434326172 % loss: 0.030356450006365776 test_acc: 50.1165771484375
Epoch: 105 / 150 Accuracy : 78.19569396972656 % loss: 0.02657872810959816 test_acc: 49.32929611206055
Epoch: 106 / 150 Accuracy : 77.65058898925781 % loss: 0.023575495928525925 test_acc: 49.177894592285156
Epoch: 107 / 150 Accuracy : 92.72281646728516 % loss: 0.03000279888510704 test_acc: 57.61090087890625
Saving Best Model with Accuracy:  99.10057067871094
Epoch: 108 / 150 Accuracy : 99.10057067871094 % loss: 0.05277499184012413 test_acc: 65.73504638671875
Epoch: 109 / 150 Accuracy : 93.32243347167969 % loss: 0.060757797211408615 test_acc: 70.57380676269531
Epoch: 110 / 150 Accuracy : 90.78768157958984 % loss: 0.03818225488066673 test_acc: 56.726722717285156
Epoch: 111 / 150 Accuracy : 98.58271789550781 % loss: 0.042159005999565125 test_acc: 63.65177917480469
Epoch: 112 / 150 Accuracy : 89.45216369628906 % loss: 0.0714535117149353 test_acc: 71.27630615234375
Epoch: 113 / 150 Accuracy : 85.44562530517578 % loss: 0.08065313845872879 test_acc: 71.52763366699219
Epoch: 114 / 150 Accuracy : 86.45407104492188 % loss: 0.07367263734340668 test_acc: 71.53368377685547
Epoch: 115 / 150 Accuracy : 85.9634780883789 % loss: 0.07297919690608978 test_acc: 71.51551818847656
Epoch: 116 / 150 Accuracy : 81.68437957763672 % loss: 0.06595765054225922 test_acc: 71.66691589355469
Epoch: 117 / 150 Accuracy : 79.85826873779297 % loss: 0.06337627023458481 test_acc: 71.68811798095703
Epoch: 118 / 150 Accuracy : 77.18724822998047 % loss: 0.05454450473189354 test_acc: 71.68811798095703
Epoch: 119 / 150 Accuracy : 76.39683532714844 % loss: 0.03190938010811806 test_acc: 71.71839141845703
Epoch: 120 / 150 Accuracy : 80.04905700683594 % loss: 0.017782112583518028 test_acc: 71.69416809082031
Epoch: 121 / 150 Accuracy : 86.23603057861328 % loss: 0.013651636429131031 test_acc: 71.57002258300781
Epoch: 122 / 150 Accuracy : 88.49822998046875 % loss: 0.013350032269954681 test_acc: 71.41256713867188
Epoch: 123 / 150 Accuracy : 94.24911499023438 % loss: 0.014440452679991722 test_acc: 70.71612548828125
Epoch: 124 / 150 Accuracy : 96.89288330078125 % loss: 0.01554773934185505 test_acc: 70.12869262695312
Epoch: 125 / 150 Accuracy : 97.76506042480469 % loss: 0.015573464334011078 test_acc: 69.71990966796875
Epoch: 126 / 150 Accuracy : 98.03761291503906 % loss: 0.015178159810602665 test_acc: 69.541259765625
Epoch: 127 / 150 Accuracy : 98.66448974609375 % loss: 0.016055122017860413 test_acc: 69.01741027832031
Saving Best Model with Accuracy:  99.23684692382812
Epoch: 128 / 150 Accuracy : 99.23684692382812 % loss: 0.017303558066487312 test_acc: 68.42391967773438
Saving Best Model with Accuracy:  99.67293548583984
Epoch: 129 / 150 Accuracy : 99.67293548583984 % loss: 0.018580619245767593 test_acc: 67.73050689697266
Saving Best Model with Accuracy:  99.7819595336914
Epoch: 130 / 150 Accuracy : 99.7819595336914 % loss: 0.019794588908553123 test_acc: 67.09765625
Saving Best Model with Accuracy:  99.80921173095703
Epoch: 131 / 150 Accuracy : 99.80921173095703 % loss: 0.02094375714659691 test_acc: 66.49810791015625
Epoch: 132 / 150 Accuracy : 99.7819595336914 % loss: 0.02233053371310234 test_acc: 65.81680297851562
Epoch: 133 / 150 Accuracy : 99.72744750976562 % loss: 0.023876212537288666 test_acc: 65.06585693359375
Epoch: 134 / 150 Accuracy : 99.61842346191406 % loss: 0.025638503953814507 test_acc: 64.17562103271484
Epoch: 135 / 150 Accuracy : 99.45489501953125 % loss: 0.02769540064036846 test_acc: 63.14610290527344
Epoch: 136 / 150 Accuracy : 99.15508270263672 % loss: 0.030203070491552353 test_acc: 61.97426223754883
Epoch: 137 / 150 Accuracy : 98.58271789550781 % loss: 0.03267187997698784 test_acc: 60.78425598144531
Epoch: 138 / 150 Accuracy : 97.65603637695312 % loss: 0.03510699048638344 test_acc: 59.73353576660156
Epoch: 139 / 150 Accuracy : 96.3477783203125 % loss: 0.03841252624988556 test_acc: 58.152915954589844
Epoch: 140 / 150 Accuracy : 94.65794372558594 % loss: 0.03289978951215744 test_acc: 56.532928466796875
Epoch: 141 / 150 Accuracy : 97.0564193725586 % loss: 0.018233314156532288 test_acc: 58.53141403198242
Epoch: 142 / 150 Accuracy : 95.44834899902344 % loss: 0.01972096785902977 test_acc: 57.311126708984375
Epoch: 143 / 150 Accuracy : 94.73970794677734 % loss: 0.02006741426885128 test_acc: 56.6056022644043
Epoch: 144 / 150 Accuracy : 94.22185516357422 % loss: 0.021008498966693878 test_acc: 56.15745544433594
Epoch: 145 / 150 Accuracy : 93.4859619140625 % loss: 0.022838827222585678 test_acc: 55.85768508911133
Epoch: 146 / 150 Accuracy : 92.36849212646484 % loss: 0.025860900059342384 test_acc: 55.33989334106445
Epoch: 147 / 150 Accuracy : 91.0329818725586 % loss: 0.030523493885993958 test_acc: 54.47993850708008
Epoch: 148 / 150 Accuracy : 99.61842346191406 % loss: 0.013472400605678558 test_acc: 67.2248306274414
Epoch: 149 / 150 Accuracy : 51.730716705322266 % loss: 0.0807848796248436 test_acc: 39.548828125
Epoch: 150 / 150 Accuracy : 87.16271209716797 % loss: 0.008421153761446476 test_acc: 52.71158218383789

Process finished with exit code 0
"""

import re
import matplotlib.pyplot as plt

# # Read the content from the specified file
log_file_path = "save_logs.txt"
with open(log_file_path, "r") as file:
     text = file.read()

# Split the text into individual sections for each graph
graph_sections = text.split("new graph starts here\n")[1:]

for idx, section in enumerate(graph_sections):
    # Extract lr, batch size, number of epochs, epoch numbers, accuracies, loss values, and test_acc values using regex
    lr_match = re.search(r"the lr is: ([\d.e-]+)", section)
    lr = lr_match.group(1)  # Keep the original LR format
    batch_size = int(re.search(r"the batch size is: ([\d.]+)", section).group(1))
    num_epochs = int(re.search(r"the number of epochs is: ([\d.]+)", section).group(1))
    data = re.findall(r"Epoch: (\d+) / \d+ Accuracy : ([\d.]+) % loss: ([\de.-]+) test_acc: ([\d.]+)", section)

    print("num_epochs:", num_epochs)
    print("len(data):", len(data))
    if num_epochs != len(data):
        print("Error: num_epochs and len(data) are not equal.")
        # sys.exit(1)

    print(data[-5:])
    # Convert extracted data to appropriate types
    epoch_numbers, accuracies, loss_values, test_acc_values = zip(*data)
    epoch_numbers = list(map(int, epoch_numbers))
    accuracies = list(map(float, accuracies))
    loss_values = list(map(float, loss_values))
    test_acc_values = list(map(float, test_acc_values))

    # Create a new figure for each plot with a wider figure size
    plt.figure(figsize=(20, 5))  # Adjust the figure size

    # Plot for Losses
    plt.subplot(1, 2, 1)
    plt.plot(epoch_numbers, loss_values, marker='o', label='Loss')
    plt.title('Losses Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # Plot for Accuracy and Test Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epoch_numbers, accuracies, marker='o', color='orange', label='Accuracy')
    plt.plot(epoch_numbers, test_acc_values, marker='o', color='green', label='Test Accuracy')
    plt.title('Accuracy and Test Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%) / Test Accuracy (%)')
    plt.grid(True)
    plt.legend()

    # Display LR as it is (no scientific notation)
    plt.suptitle(f"Parameters: LR = {lr}, Batch Size = {batch_size}, Epochs = {num_epochs}")

    # Adjust spacing between subplots and make x-axis wider
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout(pad=3.0)

    plt.show()

print("All plots displayed.")
