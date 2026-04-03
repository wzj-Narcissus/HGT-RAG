
1. 总体对比

  Overall 指标

  ┌────────────────────────┬──────────┬──────────┬──────────┬────────┬───────────┬────────┬────────┐
  │         Method         │ Recall@1 │ Recall@3 │ Recall@5 │  MRR   │ Precision │ Recall │   F1   │
  ├────────────────────────┼──────────┼──────────┼──────────┼────────┼───────────┼────────┼────────┤
  │ HGT (d256)             │   0.3756 │   0.5682 │   0.6891 │ 0.5456 │    0.3214 │ 0.4597 │ 0.3726 │
  ├────────────────────────┼──────────┼──────────┼──────────┼────────┼───────────┼────────┼────────┤
  │ Embedding-only (d4096) │   0.3756 │   0.5378 │   0.6550 │ 0.5346 │    0.1395 │ 0.9102 │ 0.2034 │
  ├────────────────────────┼──────────┼──────────┼──────────┼────────┼───────────┼────────┼────────┤
  │ Embedding-only (d256)  │   0.3673 │   0.5204 │   0.6416 │ 0.5222 │    0.1395 │ 0.9311 │ 0.2038 │
  └────────────────────────┴──────────┴──────────┴──────────┴────────┴───────────┴────────┴────────┘

  怎么解读

  - HGT 的排序能力最好：MRR / Recall@3 / Recall@5 全部最高
  - 4096 维纯 embedding 比 256 维纯 embedding 更强
  - HGT 的 precision / F1 大幅领先
  - 但 embedding-only 的 overall recall 更高，说明它更像“高召回粗排器”

  一句话：
  - 如果你想要更好的排序和最终筛选，HGT 更好
  - 如果你想要高召回候选集，embedding-only 更像召回器

  ---
  2. 各模态对比

  TextBlock

  ┌───────────────┬────────┬────────┬────────┬────────┐
  │    Method     │  R@1   │  R@5   │  MRR   │   F1   │
  ├───────────────┼────────┼────────┼────────┼────────┤
  │ HGT           │ 0.4153 │ 0.8672 │ 0.6864 │ 0.4074 │
  ├───────────────┼────────┼────────┼────────┼────────┤
  │ Emb-only 4096 │ 0.2677 │ 0.6834 │ 0.5459 │ 0.1261 │
  ├───────────────┼────────┼────────┼────────┼────────┤
  │ Emb-only 256  │ 0.2467 │ 0.6488 │ 0.5150 │ 0.1261 │
  └───────────────┴────────┴────────┴────────┴────────┘

  结论：
  HGT 在 textblock 上优势非常明显。
  这依然是 HGT 最核心的价值来源。

  ---
  Cell

  ┌───────────────┬────────┬────────┬────────┬────────┐
  │    Method     │  R@1   │  R@5   │  MRR   │   F1   │
  ├───────────────┼────────┼────────┼────────┼────────┤
  │ HGT           │ 0.0460 │ 0.2178 │ 0.1693 │ 0.0818 │
  ├───────────────┼────────┼────────┼────────┼────────┤
  │ Emb-only 4096 │ 0.1178 │ 0.3105 │ 0.2373 │ 0.0137 │
  ├───────────────┼────────┼────────┼────────┼────────┤
  │ Emb-only 256  │ 0.0886 │ 0.2685 │ 0.2024 │ 0.0136 │
  └───────────────┴────────┴────────┴────────┴────────┘

  结论：
  Cell 上 HGT 还是明显落后于纯 embedding。
  而且 4096 维比 256 维又更强，说明 cell 更依赖原始高维语义表示。

  这还是当前系统最大短板。

  ---
  Image

  ┌───────────────┬────────┬────────┬────────┬────────┐
  │    Method     │  R@1   │  R@5   │  MRR   │   F1   │
  ├───────────────┼────────┼────────┼────────┼────────┤
  │ HGT           │ 0.2041 │ 0.6759 │ 0.4328 │ 0.2459 │
  ├───────────────┼────────┼────────┼────────┼────────┤
  │ Emb-only 4096 │ 0.2063 │ 0.6426 │ 0.4168 │ 0.1051 │
  ├───────────────┼────────┼────────┼────────┼────────┤
  │ Emb-only 256  │ 0.2142 │ 0.6536 │ 0.4247 │ 0.1066 │
  └───────────────┴────────┴────────┴────────┴────────┘

  结论：
  - Image 上三者差距没 textblock 那么大
  - HGT 的 R@5 / MRR / F1 最好
  - 但 R@1 并不占优

  所以 image 上是 小幅优势，不是压倒性优势。

  ---
  Caption

  ┌───────────────┬────────┬────────┬────────┬────────┐
  │    Method     │  R@1   │  R@5   │  MRR   │   F1   │
  ├───────────────┼────────┼────────┼────────┼────────┤
  │ HGT           │ 0.2126 │ 0.6848 │ 0.4393 │ 0.2470 │
  ├───────────────┼────────┼────────┼────────┼────────┤
  │ Emb-only 4096 │ 0.2863 │ 0.6384 │ 0.4732 │ 0.0893 │
  ├───────────────┼────────┼────────┼────────┼────────┤
  │ Emb-only 256  │ 0.2872 │ 0.6369 │ 0.4691 │ 0.0896 │
  └───────────────┴────────┴────────┴────────┴────────┘

  结论：
  - caption 上排序前部其实是 embedding-only 更强
  - 但 HGT 的阈值筛选类指标更好
  - 说明 HGT 对 caption 的“最终判断”更保守

  ---
  Table

  三者基本都顶满了：

  ┌───────────────┬────────┬────────┐
  │    Method     │  R@1   │  MRR   │
  ├───────────────┼────────┼────────┤
  │ HGT           │ 1.0000 │ 1.0000 │
  ├───────────────┼────────┼────────┤
  │ Emb-only 4096 │ 1.0000 │ 1.0000 │
  ├───────────────┼────────┼────────┤
  │ Emb-only 256  │ 1.0000 │ 1.0000 │
  └───────────────┴────────┴────────┘

  结论：
  table 基本已经不是区分方法优劣的地方。

  ---
  3. HGT 相比两个 embedding baseline 的提升

  HGT vs Embedding-only 256

  - MRR: 0.5456 - 0.5222 = +0.0234
  - Recall@5: 0.6891 - 0.6416 = +0.0475
  - F1: 0.3726 - 0.2038 = +0.1688

  HGT vs Embedding-only 4096

  - MRR: 0.5456 - 0.5346 = +0.0110
  - Recall@5: 0.6891 - 0.6550 = +0.0341
  - F1: 0.3726 - 0.2034 = +0.1692

  这说明什么

  - 如果 baseline 用 256维 embedding-only，HGT 提升比较明显
  - 如果 baseline 用 4096维 embedding-only，HGT 仍然更好，但优势变小
  - 也就是说，高维原始 embedding 本身已经很强，HGT 的增益一部分被强表示能力吃掉了

  ---
  4. 4096维 vs 256维 embedding-only

  Overall

  - MRR: 0.5346 > 0.5222
  - Recall@1: 0.3756 > 0.3673
  - Recall@5: 0.6550 > 0.6416

  特别明显的提升在

  - TextBlock
  - Cell

  尤其是 cell：
  - d4096 MRR = 0.2373
  - d256 MRR = 0.2024

  这说明：
  ▎ 对纯相似度检索来说，投影到 256 维会损失一部分细粒度区分能力，尤其是 cell 这类难节点。

  ---
  5. 当前最合理的系统定位

  如果你的目标是做最终检索器

  优先选：
  - HGT

  因为它：
  - overall 排序最好
  - textblock 提升最大
  - 最终 precision / f1 明显更强

  如果你的目标是做第一阶段召回

  优先考虑：
  - Embedding-only 4096

  因为它：
  - 比 d256 更强
  - recall 很高
  - cell 反而比 HGT 好

  所以最自然的架构其实是：

  ▎ Stage 1: embedding-only 4096 召回候选
  ▎ Stage 2: HGT 重排 / 精排

  这个方向现在从结果上看很合理。

  ---
  6. 当前最值得关注的问题

  还是老问题：

  Cell

  HGT 在 cell 上仍然落后：
  - HGT cell MRR = 0.1693
  - Emb-only 4096 cell MRR = 0.2373

  这说明 HGT 的整体潜力还没完全发挥出来。
  因为它已经把 textblock 拉高了，但被 cell 拖住了。

  ---
  7. 最终结论

  一句话总结：

  ▎ 三种检索方式里，HGT 整体效果最好；4096维 embedding-only 是最强的纯 embedding baseline，明显优于 256维；HGT 的主要收益来自
  textblock 和整体精排能力，但 cell 仍然是当前瓶颈。

  ---
  8. 结果文件位置

  - HGT: /root/multimodalqa/outputs/predictions/metrics_dev.json
  - Embedding-only 256: /root/multimodalqa/outputs/predictions/embedding_only_d256_dev_metrics.json
  - Embedding-only 4096: /root/multimodalqa/outputs/predictions/embedding_only_d4096_dev_metrics.json

  ---
  如果你愿意，我下一步可以直接帮你做一张适合放进论文/汇报的三方法对比表，或者继续帮你分析：

  1. 为什么 HGT 在 textblock 上强、在 cell 上弱
  2. 如何把 4096 维 embedding-only 变成 HGT 的第一阶段召回器

✻ Cooked for 5m 24s

● Background command "Run embedding-only 4096-dim evaluation on dev split" completed (exit code 0)

● Background command "Run embedding-only 256-dim evaluation on dev split" completed (exit code 0)

● Background command "Run HGT retrieval evaluation on dev split" completed (exit code 0)

● 我读取这三次后台评测的输出，核对结果。

  Read 3 files (ctrl+o to expand)                                                                                                 
                                                                                                 
● 已读取。这里有个关键更正：                                                                                                      
                                                 
  - b59tq55et：HGT dev 评测结果正常                                                                                               
  - b04r5neqj：embedding-only 256 维 dev 评测结果正常                                                                             
  - b9y90az06：这次后台 4096 维结果是无效的，因为它读的是错误目录，实际加载了 0 个图                                              
                                                                                                                                  
  也就是这份后台文件里：                                                                                                          
                                                                                                                                  
  - /tmp/.../b9y90az06.output:1-4                                                                                                 
                  
  显示的是：                                                                                                                      
  - Loading 0 hetero data files
  - Loaded 0 graphs                                                                                                               
                   
  所以 不能用这份后台输出做 4096 维对比。                                                                                         
                                                                                                                                  
  我前面已经补跑了正确目录下的 4096 维评测，正确结果是：                                                                          
                                                                                                                                  
  - overall/recall@1 = 0.3756                                                                                                     
  - overall/recall@3 = 0.5378
  - overall/recall@5 = 0.6550                                                                                                     
  - overall/mrr = 0.5346                                                                                                          
  - overall/f1 = 0.2034                                                                                                           
                                                                                                                                  
  ---                                                                                                                             
  这三次后台输出确认到的有效结果                                                                                                  
                                                                                                                                  
  1) HGT          
                                                                                                                                  
  来自 b59tq55et.output                                                                                                           
                                                                                                                                  
  - overall/recall@1 = 0.3756                                                                                                     
  - overall/recall@3 = 0.5682
  - overall/recall@5 = 0.6891                                                                                                     
  - overall/mrr = 0.5456                                                                                                          
  - overall/precision = 0.3214                                                                                                    
  - overall/recall = 0.4597                                                                                                       
  - overall/f1 = 0.3726                                                                                                           
                                                                                                                                  
  2) Embedding-only 256                                                                                                           
                                                                                                                                  
  来自 b04r5neqj.output                                                                                                           
                  
  - overall/recall@1 = 0.3673                                                                                                     
  - overall/recall@3 = 0.5204
  - overall/recall@5 = 0.6416                                                                                                     
  - overall/mrr = 0.5222                                                                                                          
  - overall/precision = 0.1395                                                                                                    
  - overall/recall = 0.9311                                                                                                       
  - overall/f1 = 0.2038                                                                                                           
                                                                                                                                  
  3) Embedding-only 4096                                                                                                          
                                                                                                                                  
  b9y90az06.output 无效。                                                                                                         
  正确结果来自我补跑的正确目录：
                                                                                                                                  
  - overall/recall@1 = 0.3756                                                                                                     
  - overall/recall@3 = 0.5378                                                                                                     
  - overall/recall@5 = 0.6550                                                                                                     
  - overall/mrr = 0.5346                                                                                                          
  - overall/precision = 0.1395                                                                                                    
  - overall/recall = 0.9102                                                                                                       
  - overall/f1 = 0.2034                                                                                                           
                                                                                                                                  
  ---                                                                                                                             
  最终可用对比表                                                                                                                  
                                                                                                                                  
  ┌───────────────┬────────┬────────┬────────┬────────┬───────────┬────────┬────────┐
  │    Method     │  R@1   │  R@3   │  R@5   │  MRR   │ Precision │ Recall │   F1   │                                             
  ├───────────────┼────────┼────────┼────────┼────────┼───────────┼────────┼────────┤                                             
  │ HGT           │ 0.3756 │ 0.5682 │ 0.6891 │ 0.5456 │    0.3214 │ 0.4597 │ 0.3726 │                                             
  ├───────────────┼────────┼────────┼────────┼────────┼───────────┼────────┼────────┤                                             
  │ Emb-only 4096 │ 0.3756 │ 0.5378 │ 0.6550 │ 0.5346 │    0.1395 │ 0.9102 │ 0.2034 │                                             
  ├───────────────┼────────┼────────┼────────┼────────┼───────────┼────────┼────────┤                                             
  │ Emb-only 256  │ 0.3673 │ 0.5204 │ 0.6416 │ 0.5222 │    0.1395 │ 0.9311 │ 0.2038 │                                             
  └───────────────┴────────┴────────┴────────┴────────┴───────────┴────────┴────────┘                                             
                  
  ---                                                                                                                             
  结论再确认一次  
                                                                                                                                  
  - 排序最好：HGT
  - 纯 embedding 最好：4096 维                                                                                                    
  - 召回最高：256 维 embedding-only                                                                                               
  - 最终筛选质量最好：HGT                                                                                                         
                                                                                                                                  
  所以最合理的系统定位仍然是：                                                                                                    
                                                                                                                                  
  ▎ Embedding-only 4096 做召回，HGT 做重排。                                                                                      
                  
  如果你要，我下一步可以把这三组结果整理成一段可以直接写进论文/实验报告的中文分析。我该如何优化HGT效果

cell_all
  === Evaluation Results ===
  caption/f1: 0.2434
  caption/mrr: 0.4426
  caption/precision: 0.1541
  caption/recall: 0.5785
  caption/recall@1: 0.2176
  caption/recall@3: 0.4925
  caption/recall@5: 0.6791
  cell/f1: 0.0573
  cell/mrr: 0.1765
  cell/precision: 0.0359
  cell/recall: 0.1418
  cell/recall@1: 0.0532
  cell/recall@3: 0.1429
  cell/recall@5: 0.2273
  image/f1: 0.2420
  image/mrr: 0.4408
  image/precision: 0.1755
  image/recall: 0.3895
  image/recall@1: 0.2165
  image/recall@3: 0.4798
  image/recall@5: 0.6890
  loss: 0.4959
  overall/f1: 0.3678
  overall/mrr: 0.5485
  overall/precision: 0.3121
  overall/recall: 0.5065
  overall/recall@1: 0.3795
  overall/recall@3: 0.5664
  overall/recall@5: 0.6940
  table/f1: 0.8784
  table/mrr: 1.0000
  table/precision: 0.8313
  table/recall: 0.9312
  table/recall@1: 1.0000
  table/recall@3: 1.0000
  table/recall@5: 1.0000
  textblock/f1: 0.4179
  textblock/mrr: 0.6828
  textblock/precision: 0.3637
  textblock/recall: 0.4912
  textblock/recall@1: 0.4100
  textblock/recall@3: 0.7167
  textblock/recall@5: 0.8746

  
  top20
=== Evaluation Results ===
  caption/f1: 0.2343
  caption/mrr: 0.4219
  caption/precision: 0.1528
  caption/recall: 0.5022
  caption/recall@1: 0.1941
  caption/recall@3: 0.4664
  caption/recall@5: 0.6668
  cell/f1: 0.0416
  cell/mrr: 0.1367
  cell/precision: 0.0227
  cell/recall: 0.2464
  cell/recall@1: 0.0295
  cell/recall@3: 0.0858
  cell/recall@5: 0.1558
  image/f1: 0.2343
  image/mrr: 0.4182
  image/precision: 0.1544
  image/recall: 0.4862
  image/recall@1: 0.1879
  image/recall@3: 0.4650
  image/recall@5: 0.6575
  loss: 0.4981
  overall/f1: 0.3608
  overall/mrr: 0.5317
  overall/precision: 0.3105
  overall/recall: 0.5219
  overall/recall@1: 0.3642
  overall/recall@3: 0.5446
  overall/recall@5: 0.6690
  table/f1: 0.8850
  table/mrr: 1.0000
  table/precision: 0.8246
  table/recall: 0.9549
  table/recall@1: 1.0000
  table/recall@3: 1.0000
  table/recall@5: 1.0000
  textblock/f1: 0.4088
  textblock/mrr: 0.6818
  textblock/precision: 0.3982
  textblock/recall: 0.4200
  textblock/recall@1: 0.4095
  textblock/recall@3: 0.7056
  textblock/recall@5: 0.8647

  hybrid20
  === Evaluation Results ===
  caption/f1: 0.2430
  caption/mrr: 0.4167
  caption/precision: 0.1667
  caption/recall: 0.4481
  caption/recall@1: 0.1819
  caption/recall@3: 0.4647
  caption/recall@5: 0.6880
  cell/f1: 0.0566
  cell/mrr: 0.1669
  cell/precision: 0.0326
  cell/recall: 0.2156
  cell/recall@1: 0.0457
  cell/recall@3: 0.1337
  cell/recall@5: 0.2136
  image/f1: 0.2472
  image/mrr: 0.4235
  image/precision: 0.2077
  image/recall: 0.3052
  image/recall@1: 0.1913
  image/recall@3: 0.4703
  image/recall@5: 0.6743
  loss: 0.5114
  overall/f1: 0.3679
  overall/mrr: 0.5388
  overall/precision: 0.3148
  overall/recall: 0.4900
  overall/recall@1: 0.3677
  overall/recall@3: 0.5559
  overall/recall@5: 0.6878
  table/f1: 0.8820
  table/mrr: 1.0000
  table/precision: 0.8409
  table/recall: 0.9273
  table/recall@1: 1.0000
  table/recall@3: 1.0000
  table/recall@5: 1.0000
  textblock/f1: 0.4106
  textblock/mrr: 0.6870
  textblock/precision: 0.3263
  textblock/recall: 0.5539
  textblock/recall@1: 0.4198
  textblock/recall@3: 0.7109
  textblock/recall@5: 0.8631


top10
  === Evaluation Results ===
  caption/f1: 0.2431
  caption/mrr: 0.4164
  caption/precision: 0.1619
  caption/recall: 0.4880
  caption/recall@1: 0.1862
  caption/recall@3: 0.4423
  caption/recall@5: 0.6685
  cell/f1: 0.0354
  cell/mrr: 0.1128
  cell/precision: 0.0187
  cell/recall: 0.3436
  cell/recall@1: 0.0186
  cell/recall@3: 0.0699
  cell/recall@5: 0.1179
  image/f1: 0.2433
  image/mrr: 0.4274
  image/precision: 0.1878
  image/recall: 0.3452
  image/recall@1: 0.2032
  image/recall@3: 0.4571
  image/recall@5: 0.6644
  loss: 0.5121
  overall/f1: 0.3632
  overall/mrr: 0.5282
  overall/precision: 0.3183
  overall/recall: 0.5101
  overall/recall@1: 0.3646
  overall/recall@3: 0.5365
  overall/recall@5: 0.6642
  table/f1: 0.8807
  table/mrr: 1.0000
  table/precision: 0.8266
  table/recall: 0.9423
  table/recall@1: 1.0000
  table/recall@3: 1.0000
  table/recall@5: 1.0000
  textblock/f1: 0.4133
  textblock/mrr: 0.6843
  textblock/precision: 0.3965
  textblock/recall: 0.4315
  textblock/recall@1: 0.4149
  textblock/recall@3: 0.7134
  textblock/recall@5: 0.8701