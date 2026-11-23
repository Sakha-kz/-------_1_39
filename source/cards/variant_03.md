№3. CIFAR-10 + ViT
Суть: CIFAR-10, ViT-tiny (timm) scratch и transfer; фокус — «голодность»
трансформеров до данных. Вопрос: правда ли ViT «голоднее до данных» — что
происходит на 50k CIFAR? : attention rollout вместо Grad-CAM (для ViT).

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1003. cifar10. EDA + contact
sheet. 2. Классика (ст. 2). Лист обработки. SIFT-BoW (64 слова) + LinearSVC: --
classical siftbow (словарь fit на train). Работают ли локальные особенности на
32×32? 3. Аугментации (ст. 3). strong (RandAugment). Абляция none vs strong. 4.
Модель (ст. 4). vit_tiny (timm vit_tiny_patch16_224). Params/память; transfer — --
pretrained. 5. Обучение (ст. 5). AdamW lr 3e-4, 40 эпох. Scratch и transfer
(probe→FT, блок 9). Кривые, resume. 6. Оценка (ст. 6). Полная + robustness;
сравни с CNN (№1/№2) при равном бюджете. 7. Grad-CAM (ст. 7). Для ViT —
attention rollout (бонус) или CAM по patch-embedding; 5+5. 8. Перенос (ст. 8).
transfer ViT: probe→FT; таблица scratch vs probe vs FT и сравнение с CNN. 9.

Отчёт (ст. 9). Latency, cost, ответ на вопрос (data hunger), таблица
классика/scratch/transfer.

Команды:

python train.py --dataset cifar10 --model vit_tiny --aug strong --optimizer adamw --lr 3e-4 \
 --scheduler cosine --epochs 40 --size 224 --seed 1003 --out_dir drive/out-var3-scratch
python evaluate.py --ckpt drive/out-var3-scratch/best.pt --dataset cifar10 --model vit_tiny \
 --size 224 --split val --seed 1003 --classical siftbow --processing --gradcam --robustness --
latency --out_dir drive/eval-var3

Артефакты: кривые ViT scratch vs transfer, attention rollout, сравнение с CNN,
отчёт. Ловушка: ViT scratch на 50k без strong-аугментаций сильно
переобучается — не делай вывод о «голодности» без абляции аугментаций.

