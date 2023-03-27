cascade -ner  使用了两种方法，一种是span、一种是序列标注



**序列标注+cls：**

![image-20230326234856610](/home/huangkai/.config/Typora/typora-user-images/image-20230326234856610.png)

label1：[BIEO]

label2:   [cls1,cls1,cls1,o,o,cls2,cls2,...] 

model: 1.先预测BIEO，对于BIE标签，抽取embedding1及index，送入ffn，获得对应的label。

**span+cls：**

图来自于2020A Novel Cascade Binary Tagging Framework for Relational Triple Extraction 关系抽取论文，第二步改为dense-cls层即可。

![image-20230327003010301](/home/huangkai/.config/Typora/typora-user-images/image-20230327003010301.png)

label1：[(head_label1，tail_label1),(head_label2，tail_label2)...]

label2:   [cls1,cls1,cls1,o,o,cls2,cls2,...] 

model: 1.获得实体的头部指针和尾部指针;2.抽取指针及index，分别送入ffn，得到对应的类型。





整体的差异在于第一步，分别用了序列标注方法和span抽取方法。