from openai import OpenAI
import pandas as pd
from tqdm import tqdm

client = OpenAI(
    api_key="",
    base_url="",
)

lines = open("违规词.txt", "r", encoding="utf-8").readlines()
illegalWord_list = []
for line in lines:
    illegalWord_list.extend(line.split('、'))


def deepseekResponse(target_words):
    prompt_sys = f"""你是一个电商直播推销员，请仿照下面例子，给出一个350字左右的保健品直播内容，只需要介绍产品及功效，不要包含任何开头语和结束语如大家好等，内容必须包含以下句子：{target_words}。"""
    prompt_text = f"""####例子1####
出现到白板上面，其中一种情况，宝宝一定得要去重视起来的。因为它这个就是你小心肝不堪重负，身体在皮心里给你发的那个信号。像平常呀经常这样子的话呢，宝宝你不去管它，日积月累下来它就能滚雪球胺的，很容易逐渐去出现一些不可逆转的问题的啊。宝宝。那么像我们家一号链接这款精华片，顾名思义就是帮助你把这么多年熬的夜喝的久。对于在我们刚刚拉圾给大家做一个减法精华的这一款，它还是明星情男同款。在吃降温精华片，即这款肝片也是款胃片二合一一瓶搞定。管到两个方面的，都不需要你在单独去拍胃片的。尤其这款干片，不仅明星秦男在吃，我自己包括爸妈，很多宝宝它都是有在吃药的，因为你很多都它都是做不到。晚上十一点之前，所以叫吗？你超过。
####例子2####
朋友们，所有的产品呢都是给你经过了正规的普逆险的报告，给你做到了持证上岗里面的重金属检测、微生物检测、有害物质检测含量检测呢是严格符合标准的。你可以放胆去吃。而且诺泰兰德咱告诉大家，不是什么小作坊小品牌啊，连续三年全国全网膳食养品销量第一的。如果说东西不好，或者说质量不行的话，咱是不可能拿到这样的一个成绩的。你别说拜保三年的朋友们三天都费劲，对不对？但是既然能够连续霸保三年，也就说明到了大多数人他选择来选择去，最终还是选择诺泰兰德啊，让咱买的安心吃的更加放心一些。朋友对，现在就拍，现在去带七号，可以的，没有问题，三岁以上咱都能去吃啊，三岁以上的都能去自动去带了维生素b朋友们既然今天呢之所以要花钱买着去吃，因为我们的人体自己是没有办法去合成维生素b的，只能够通过外界的渠道去获得。
####例子3####
都在学习，对不对？再学再背，再记，都在用到大脑，你希望孩子更好更优秀。你希望孩子了解到体面劳发，营养就不能给孩子落下。你希望马儿跑，你还得给马儿吃草呢。你希望孩子更好更优秀，你也得给孩子带到营养呀。二号链接来最后的两单了啊，二号链接给大家做到了一百毫克纯DHA,一百毫克，纯DRA含量高纯度高。它没有枣油味，没有鱼腥味儿啊，爆浆的甜橙口味啊，孩子非常爱吃，我之前买的都不送钙，今天好划算，拍了。好的，来给布灵布灵安排加急来二和链接抢光了，已经没有了。刚刚的订单我就去给大家安排发货了。你们没有买到的姐姐以后再来试吃，我也是不送的，钙片咱也是没有的，因为本身它钙片就是额外送的。如果今天不是七七专场，钙钙片肯定是送不了，试吃，肯定是没有的。而且刚刚给大家都发了券，你刚刚的价格是多少钱呀？二百四十九呀，现在你去看二百六十九呀，你知不知道姐姐你不给孩子带DHA,你不给孩子吃大脑营养，那总有家长给孩子吃，那人和人的差距就那就拉开了呀。
####回答####
"""
    completion = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {'role': 'system', 'content': prompt_sys},
            {'role': 'user', 'content': prompt_text}
        ],temperature=1.4
    )
    return completion.choices[0].message.content
illegalWord_list = ["销量第一"]
for illegal in tqdm(illegalWord_list):
    print(illegal)
    print('-----------')
    for i in tqdm(range(1)):
        write_data = deepseekResponse(illegal)
        print(write_data)
        write_data_final = []
        tmp_list = ['LLM构造',write_data,1,illegal]
        write_data_final.append(tmp_list)
        # Excel文件路径
        excel_file_path = 'LLM_Construct.xlsx'
        txt_file_path = "response.txt"
        with open(txt_file_path, 'a', encoding='utf-8') as file:
            for row in write_data_final:
                print(row[1])
                print('------'*20)
                line = "LLM构造\t" + row[1] + "\t" + illegal
                file.write(f"8-28\t{line}\n")
        # 加载现有的Excel文件，如果不存在则创建一个新的工作簿
        # 将多列数据转换为DataFrame
        new_df = pd.DataFrame(write_data_final, columns=['来源','内容', '是否违规', '违规话语'])
        with pd.ExcelWriter(excel_file_path, mode='a', if_sheet_exists='overlay') as writer:
            new_df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
