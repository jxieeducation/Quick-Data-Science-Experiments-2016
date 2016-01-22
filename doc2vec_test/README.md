http://sifaka.cs.uiuc.edu/~wang296/Data/index.html

cat amazon_mp3 | grep "\[fullText\]:" | sed 's/\[fullText\]://' > refined_mp3
cat amazon_mp3 | grep "\[productName\]:" | sed 's/\[productName\]://' > refined_product_titles



