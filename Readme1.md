### 创建环境
由于需要加载预训练模型需要
git lfs install
git clone https://huggingface.co/bert-base-chinese（可以换成其他的模型）
然后修改arg_parser.add_argument('--pre_path', default='bert-base-chinese')里的default
以及 utils\example中.pycls.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese') 括号里的模型名称
### 运行
加载完预训练模型之后 
在根目录下运行
    python scripts/slu_baseline.py
可以完成训练的过程
    python scripts\test.py
可以完成输出测试集
