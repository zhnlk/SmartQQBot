SmartQQ-Bot Ver-0.2
=========
**注意:**此框架现已基本稳定，文档会尽快完善。

***该分支使用二维码登陆的协议参考了[原名：SmartQQ-for-Raspberry-Pi(PiWebQQV2)](https://github.com/xqin/SmartQQ-for-Raspberry-Pi)这一项目***，自行整合开发了基于SmartQQ的自动机器人**框架**。

登陆时采用QQ安全中心的二维码做为登陆条件, 不需要在程序里输入QQ号码及QQ密码。首次登陆后将在本地保存cookies，下次启动机器人会尝试自动登录。

*旧版单文件机器人仍有保留在old_QQBot.py中。*

## 依赖
+ `PIL`

##快速开始
+ `python main.py`
+ 等待弹出二维码进行扫描登陆，或手动打开脚本所在目录的v.jpg进行扫描。
+ 等待登陆成功的提示
+ 各功能的启用，需要修改config文件夹下的共有设置QQBot_default.conf中进行功能的开启关闭。
+ 首次登陆过后，以后的登陆会尝试使用保存的cookie进行自动登录。

##基本功能

+ 群聊功能：
<small>注：以下命令皆是在qq中发送，群聊命令发送到所在群中</small>

	+ 群聊吐槽功能(tucao)，类似于小黄鸡，在群中通过发送`!learn {ha}{哈哈}`语句，则机器人检测到发言中包含“ha”时将自动回复“哈哈”。`!delete {ha}{哈哈}`可以删除该内容。吐槽内容本地保存在data/tucao_save/中。

	+ 群聊复读功能(repeat)，检测到群聊中***连续两个***回复内容相同，将自动复读该内容1次。

	+ 群聊关注功能(follow)，使用命令`!follow qq号`可以使机器人复读此人所有发言（除命令外）使用命令`!unfollow qq号`解除关注。qq号处可使用"me"来快速关注与解除关注自己，例：`!follow me`

	+ 群聊唤出功能(callout)，群聊中检测关键词`智障机器人`，若发言中包含该词，将自动回复`干嘛（‘·д·）`，此功能一般用于检测机器人状态与调戏

	+ 群聊命令功能(command_0arg/command_1arg)：使用`![命令名]`格式或`![命令名] {参数1}`执行命令，命令“吐槽列表”，使用命令`!吐槽列表`在群聊中激活，列出当前群的吐槽列表。
		+ 现有无参数命令：
			+ `!吐槽列表`:列出当前群的吐槽列表
		+ 现有单参数命令:
			+ `!删除关键字 {blablabla}`:删除关键字“blablabla”下的所有吐槽内容

+ 私聊功能：
	+ 私聊唤出功能(callout)，私聊中检测关键词`智障机器人`，若发言中包含该词，将自动回复`干嘛（‘·д·）`，此功能一般用于检测机器人状态与调戏

	+ 私聊复读功能(repeat)，检测到私聊中***连续两个***回复内容相同，将自动复读该内容1次。

+ 临时对话功能：
	+ 唤出功能(callout)，具体同私聊与群聊。


+ 集成的二次开发功能
1. 天气查询
命令:	`天气 上海` 或者` weather 上海` 或者`天气 shanghai` 或者` weather shanghai`

2. 图灵问答插件
命令： `ask 你的问题` 或者`问 你是谁`

3. 谁是卧底
开始游戏： `!game 谁是卧底5人局`
结束游戏： `!game end`

##了解更多细节请查看Wiki：

+ [SmartQQ消息协议封装说明](https://github.com/Yinzo/SmartQQBot/wiki/SmartQQ%E6%B6%88%E6%81%AF%E5%8D%8F%E8%AE%AE%E5%B0%81%E8%A3%85%E8%AF%B4%E6%98%8E)
+ [如何二次开发自定义功能](https://github.com/Yinzo/SmartQQBot/wiki/%E5%A6%82%E4%BD%95%E4%BA%8C%E6%AC%A1%E5%BC%80%E5%8F%91%E8%87%AA%E5%AE%9A%E4%B9%89%E5%8A%9F%E8%83%BD)
+ [程序架构逻辑](https://github.com/Yinzo/SmartQQBot/wiki/%E7%A8%8B%E5%BA%8F%E6%9E%B6%E6%9E%84%E9%80%BB%E8%BE%91)


##TODO

+ 添加群聊吐槽字数限制
+ 回复语句外置便于修改
+ 开发命令控制模块
+ 编写文档
+ Friend类补充
+ 尽可能地简化二次开发的复杂性
+ 寻找偶尔被保护的原因
+ 短时间程序断线 不需要重新扫描二维码，可以直接登录

##账户被保护的可能原因：
+ 多次发言中包含网址
+ 短时间内多次发言中包含敏感词汇
+ 短时间多次发送相同内容
+ 短时间异地登陆




