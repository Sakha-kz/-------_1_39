№15. Flowers102 + ViT-tiny transfer
Суть: Flowers102 (тот же subset 20, что у №14), ViT-tiny transfer; фокус — CNN vs
ViT на fine-grained. Вопрос: кто лучше на мелких различиях при равном
бюджете обучения? : сравнить latency обеих моделей.

Что делать по стадиям: 1. Данные (ст. 0–1). Seed 1015. flowers102, тот же
subset 20 классов, что у №14. EDA + contact sheet. 2. Классика (ст. 2). Лист
обработки. colorhist + HOG (concat) + LinearSVC: код блока 4, np.concatenate
двух признаков. Комбинация цвет+форма. 3. Аугментации (ст. 3). strong.
Абляция none vs strong. 4. Модель (ст. 4). vit_tiny + ImageNet (--pretrained).
Params/память. 5. Обучение (ст. 5). AdamW lr 3e-4, transfer probe→FT (блок 9), --
norm imagenet --size 224. Кривые, resume. 6. Оценка (ст. 6). Полная + robustness;
сравни с №14 (CNN) при равном бюджете. 7. Grad-CAM (ст. 7). Attention rollout
для ViT; 5+5. 8. Перенос (ст. 8). Сравнение с №14: таблица CNN vs ViT (acc,
macro-F1, время, params). 9. Отчёт (ст. 9). Latency обеих моделей (бонус), cost,
ответ на вопрос (кто лучше), таблица классика/CNN/ViT.

Команды:

python train.py --dataset flowers102 --model vit_tiny --pretrained --aug strong --norm
imagenet \
 --size 224 --optimizer adamw --lr 3e-4 --scheduler cosine --epochs 10 --seed 1015 --out_dir
drive/out-var15-ft
python evaluate.py --ckpt drive/out-var15-ft/best.pt --dataset flowers102 --model vit_tiny \
 --norm imagenet --size 224 --split val --seed 1015 --processing --gradcam --robustness --
latency --out_dir drive/eval-var15

Артефакты: таблица CNN vs ViT, latency обеих, concat-классика (код), отчёт.
Ловушка: concat colorhist+HOG через --classical не сделать — код блока 4; и
сравнивать с №14 на другом subset нельзя.

