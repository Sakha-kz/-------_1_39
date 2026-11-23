# Покрытие вариантов

Код подготовлен для запусков. Полное обучение ещё не выполнено. Бонусные и визуальные задания не считаются автоматически завершёнными.

| № | Датасет | Модель | Спецэксперимент |
|---:|---|---|---|

| 1 | cifar10 | smallcnn | transfer |

| 2 | cifar10 | resnet18 | unfreeze |

| 3 | cifar10 | vit_tiny | transfer |

| 4 | cifar100 | resnet20 | transfer |

| 5 | mnist | lenet | domain:kmnist |

| 6 | mnist | mlp | fewshot:10 |

| 7 | fashion_mnist | smallcnn | domain:mnist |

| 8 | fashion_mnist | resnet18 | transfer |

| 9 | kmnist | smallcnn | domain:emnist |

| 10 | emnist | lenet | imbalance |

| 11 | svhn | resnet20 | domain:mnist |

| 12 | svhn | smallcnn | robust_ft |

| 13 | stl10 | resnet18 | fewshot:50 |

| 14 | flowers102 | resnet18 | transfer |

| 15 | flowers102 | vit_tiny | transfer |

| 16 | dtd | smallcnn | texture_shift |

| 17 | eurosat | resnet18 | domain:ucmerced |

| 18 | eurosat | smallcnn | transfer |

| 19 | ucmerced | resnet18 | domain:eurosat |

| 20 | oxford_pets | mobilenet_v3 | fewshot:10 |

| 21 | oxford_pets | densenet121 | transfer |

| 22 | food101 | efficientnet_b0 | transfer |

| 23 | food101 | resnet18 | transfer |

| 24 | caltech101 | resnet18 | transfer |

| 25 | quickdraw | smallcnn | domain:emnist |

| 26 | gtsrb | resnet20 | robust_ft |

| 27 | gtsrb | mobilenet_v3 | transfer |

| 28 | utkface | resnet18 | balanced_ft |

| 29 | synthetic | smallcnn | curriculum |

| 30 | synthetic | mlp | invariance |

| 31 | cifar10 | smallcnn | augmentation_sweep |

| 32 | cifar10 | resnet18 | optimizer_sweep |

| 33 | cifar10 | resnet18 | scheduler_sweep |

| 34 | cifar10 | resnet18 | regularization |

| 35 | cifar10 | resnet18 | longtail |

| 36 | cifar10 | smallcnn | distillation |

| 37 | cifar10 | resnet18 | compression |

| 38 | cifar10 | resnet18 | ssl |

| 39 | cifar10 | efficientnet_b0 | capstone |

## Дополнительные условия, которые требуют проверки

- №4: внутри-суперклассовый анализ требует отдельной fine-головы; текущая модель coarse-классификатор.

- №5: ансамбль из трёх моделей — бонус, не включён в базовый автоматический сценарий.

- №6: выравниваются параметры MLP и LeNet; картинки весов первого слоя/HOG требуют отдельного добавления.

- №8: probe по layer2/3/4 — бонус, текущий probe использует стандартную последнюю голову.

- №13: pseudo-label STL-10 unlabeled — бонус, текущий few-shot использует только размеченный train.

- №17/19: классы аэродатасетов не совпадают; перенос признаков с новой головой, а не фиктивный zero-shot.

- №16: адаптация DTD на трёх синтетических текстурах и GAP/flatten реализованы; требуется визуальный анализ.

- №20: CPU+GPU latency измеряется; few-shot k=10 выбран явно, поскольку k в карточке не задан.

- №24: общая ECE есть, дополнительная class-wise calibration требует анализа отдельно.

- №26: coverage/accuracy сохраняется; перечень опасных пар выбирается по смыслу знаков вручную.

- №27: обязательное сравнение с №26 возможно после обоих запусков; INT8-бонус отдельно от №37 не запускается.

- №29/30: новый цвет и shift/scale реализованы; heatmap позиции/rotation — дополнительные измерения.

- №32: два LR на оптимизатор; Lion/LAMB — бонус, не включён.

- №33: три scheduler реализованы; warmup-абляция — бонус.

- №34: четыре регуляризатора и SVM C-sweep реализованы; сочетание двух лучших — бонус.

- №35: weighted, sampling, focal реализованы; DRW — бонус.

- №37: физический pruning/INT8 четыре конфигурации; recovery fine-tune — дополнительный бонус.

- №39: общий бюджет может не позволить все компоненты; неполнота отражается в budget.json.

- Всем: визуальные причины ошибок, индивидуальный вопрос и межвариантные сравнения должны опираться на полученные файлы.