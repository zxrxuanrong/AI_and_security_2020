#
```
JavaScript網頁設計與TensorFlow.js人工智慧應用教本 
作者： 陳會安	
書號： AEL023900	    出版日： 2020/09/24
ISBN： 9789865026103	    EAN： 9789865026103
http://books.gotop.com.tw/v_AEL023900#07
```

# XAMPP
```

C:\xampp2\htdocs\TF


http://127.0.0.1/tf/Ch14_5_4a.html

notepad++
```

# tf_js_1.html
```
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Ch14_4_1.html</title>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
</head>
<body>
<script>
const x = tf.scalar(10.5);
document.write("張量x值: " + x + "<br/>");
document.write("等級: " + x.rank + "<br/>");
document.write("形狀: [" + x.shape + "]<br/>");
document.write("型態: " + x.dtype + "<br/>");
document.write("----------------------------------------<br/>");
const y = tf.scalar(10, "int32");
document.write("張量y值: " + y + "<br/>");
document.write("型態: " + y.dtype + "<br/>");
</script>
</body>
</html>
```

#

```
<!DOCTYPE html>
<html>


<head>
<meta charset="utf-8"/>
<title>Ch14_4_1.html</title>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
</head>


<body>


<script>
把程式放在這裡......
</script>

</body>
</html>

```
```
const x = tf.scalar(10.5);
document.write("張量x值: " + x + "<br/>");
document.write("等級: " + x.rank + "<br/>");
document.write("形狀: [" + x.shape + "]<br/>");
document.write("型態: " + x.dtype + "<br/>");
document.write("----------------------------------------<br/>");
const y = tf.scalar(10, "int32");
document.write("張量y值: " + y + "<br/>");
document.write("型態: " + y.dtype + "<br/>");
```

# 作業
```
張量x值: Tensor 10.5
張量的屬性
等級rank: 0
形狀shape: []
資料型態(data type): float32
----------------------------------------
張量y值: Tensor 10
資料型態(data type): int32
```

```
<script>
const x = tf.scalar(10.5);
document.write("張量x值: " + x + "<br/>");
document.write("張量的屬性" + "<br/>");
document.write("等級rank: " + x.rank + "<br/>");
document.write("形狀shape: [" + x.shape + "]<br/>");
document.write("資料型態(data type): " + x.dtype + "<br/>");
document.write("----------------------------------------<br/>");
const y = tf.scalar(10, "int32");
document.write("張量y值: " + y + "<br/>");
document.write("資料型態(data type): " + y.dtype + "<br/>");
</script>

```

# 折線圖
```
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Ch14_5_1.html</title>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest"></script>
</head>
<body>
<script>
async function run() { 
  let celsius = [{x: 1, y: 25.6}, {x: 3, y: 23.2}, {x: 5, y: 18.5}, {x: 7, y: 28.3},
                 {x: 9, y: 26.5}, {x: 11,y: 30.5}, {x:13, y: 32.6}, {x:15, y: 33.1}];
  tfvis.render.linechart(
    { name: '住家氣溫', tab: "折線圖" }, 
    { values: [celsius], series: ['Home'] }, 
    { xLabel: '日數', yLabel: '攝氏溫度', 
      height: 300, width: 400 }
  );
}  
run();
</script>
</body>
</html>


```
