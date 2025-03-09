# 下载旧版本的bivpois包
download.file("https://cran.r-project.org/src/contrib/Archive/bivpois/bivpois_0.50-2.tar.gz", 
              destfile = "bivpois_0.50-2.tar.gz")

# 安装下载的包
install.packages("bivpois_0.50-2.tar.gz", repos = NULL, type = "source")