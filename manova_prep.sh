#cat /home/sarfaraz/ASVSpoof_2019_expts/data/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt | grep bonafide | sort -R  | tail -101 > /home/sarfaraz/ASVSpoof_2019_expts/data/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.eval_manova10.trl.txt

#cat /home/sarfaraz/ASVSpoof_2019_expts/data/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt | grep spoof | sort -R  | tail -201 >> /home/sarfaraz/ASVSpoof_2019_expts/data/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.eval_manova10.trl.txt

python3 test.py -m /home/sarfaraz/ASVSpoof_2019_expts/resnet_models/CQCC/PA/softmax/ -l softmax --gpu 1
