# How are experiments going?

## 21.09.2018, commit: e2c5591
This happened on 'easy' data set (only +, 1, 0):
```
Epoch: 189. Loss 0.6931746207475662. Accuracy 0.524.
Epoch: 190. Loss 0.6931785525083541. Accuracy 0.524.
Epoch: 191. Loss 0.6931529085636139. Accuracy 0.523.
Epoch: 192. Loss 0.6931049168109894. Accuracy 0.525.
Epoch: 193. Loss 0.6930894664525986. Accuracy 0.523.
Epoch: 194. Loss 0.6930556814670563. Accuracy 0.524.
Epoch: 195. Loss 0.693049041569233. Accuracy 0.524.
Epoch: 196. Loss 0.6930273432135582. Accuracy 0.524.
Epoch: 197. Loss 0.6929906914234162. Accuracy 0.525.
Epoch: 198. Loss 0.6929428712129593. Accuracy 0.526.
Epoch: 199. Loss 0.6928479674458504. Accuracy 0.525.
Epoch: 200. Loss 0.6927738657593727. Accuracy 0.524.
Epoch: 201. Loss 0.692718220949173. Accuracy 0.521.
Epoch: 202. Loss 0.6926093851327896. Accuracy 0.526.
Epoch: 203. Loss 0.692521602332592. Accuracy 0.525.
Epoch: 204. Loss 0.6923780160546302. Accuracy 0.524.
Epoch: 205. Loss 0.6922290828227997. Accuracy 0.53.
Epoch: 206. Loss 0.6920813990831375. Accuracy 0.529.
Epoch: 207. Loss 0.6918943821191788. Accuracy 0.539.
Epoch: 208. Loss 0.6915499917864799. Accuracy 0.537.
Epoch: 209. Loss 0.6913969677090644. Accuracy 0.537.
Epoch: 210. Loss 0.6908350767791271. Accuracy 0.543.
Epoch: 211. Loss 0.689408360093832. Accuracy 0.547.
Epoch: 212. Loss 0.6867991574406623. Accuracy 0.559.
Epoch: 213. Loss 0.6841140533983707. Accuracy 0.566.
Epoch: 214. Loss 0.6779222503900528. Accuracy 0.577.
Epoch: 215. Loss 0.6500520876049996. Accuracy 0.62.
Epoch: 216. Loss 0.5270058020353318. Accuracy 0.755.
Epoch: 217. Loss 0.584631164520979. Accuracy 0.694.
Epoch: 218. Loss 0.45363961681723597. Accuracy 0.768.
Epoch: 219. Loss 0.008585449606180191. Accuracy 1.0.
Epoch: 220. Loss 0.0012513085603713988. Accuracy 1.0.
Epoch: 221. Loss 0.0006681718826293945. Accuracy 1.0.
Epoch: 222. Loss 0.00044301843643188476. Accuracy 1.0.
Epoch: 223. Loss 0.000326251745223999. Accuracy 1.0.
Epoch: 224. Loss 0.00025533819198608396. Accuracy 1.0.
Epoch: 225. Loss 0.00020813655853271486. Accuracy 1.0.
Epoch: 226. Loss 0.00017450404167175293. Accuracy 1.0.
Epoch: 227. Loss 0.00014956307411193847. Accuracy 1.0.
```

The stats are on training set; the first working version of treeNN.
Similar situation on the more difficult dataset (with +, -, 0, 1, 2):
```
Epoch: 70. Loss 0.6921160946816206. Accuracy 0.514.
Epoch: 71. Loss 0.6918926461488009. Accuracy 0.5153.
Epoch: 72. Loss 0.6916303490310908. Accuracy 0.5176.
Epoch: 73. Loss 0.6913758839130402. Accuracy 0.5142.
Epoch: 74. Loss 0.6907735283702612. Accuracy 0.5206.
Epoch: 75. Loss 0.6903210225135088. Accuracy 0.5209.
Epoch: 76. Loss 0.6898594205617905. Accuracy 0.5245.
Epoch: 77. Loss 0.6891386212795972. Accuracy 0.5285.
Epoch: 78. Loss 0.688923718714714. Accuracy 0.5284.
Epoch: 79. Loss 0.688395893445611. Accuracy 0.5297.
Epoch: 80. Loss 0.6873950962007046. Accuracy 0.5277.
Epoch: 81. Loss 0.6872827852964402. Accuracy 0.5309.
Epoch: 82. Loss 0.6870496775388718. Accuracy 0.5293.
Epoch: 83. Loss 0.6863596347272396. Accuracy 0.5318.
Epoch: 84. Loss 0.684767290776968. Accuracy 0.5369.
Epoch: 85. Loss 0.6844566337704658. Accuracy 0.5362.
Epoch: 86. Loss 0.6842694052070379. Accuracy 0.5431.
Epoch: 87. Loss 0.6816816687375307. Accuracy 0.5476.
Epoch: 88. Loss 0.6765236690372228. Accuracy 0.553.
Epoch: 89. Loss 0.3924415097892284. Accuracy 0.7581.
Epoch: 90. Loss 8.710682392120362e-05. Accuracy 1.0.
```


And here you go. On the "difficult" data set (+, -, 0, 1, 2) we have perfect
accuracy both on valid and test. [Log.](https://github.com/BartoszPiotrowski/deep-parity/logs/log-train-valid-difficult-set.txt)
