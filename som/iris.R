library(kohonen)

data(wines)
set.seed(7)
kohmap <- xyf(scale(wines), classvec2classmat(wine.classes),
              grid = somgrid(5, 5, "hexagonal"), rlen=100)

plot(kohmap, type="changes")
plot(kohmap, type="codes", main = c("Codes X", "Codes Y"))
plot(kohmap, type="counts")
## palette suggested by Leo Lopes
coolBlueHotRed <- function(n, alpha = 1) {
  rainbow(n, end=4/6, alpha=alpha)[n:1]
}
plot(kohmap, type="quality", palette.name = coolBlueHotRed)
plot(kohmap, type="mapping",
     labels = wine.classes, col = wine.classes+1,
     main = "mapping plot")
## add background colors to units according to their predicted class labels
xyfpredictions <- classmat2classvec(predict(kohmap)$unit.predictions)
bgcols <- c("gray", "pink", "lightgreen")
plot(kohmap, type="mapping", col = wine.classes+1,
     pchs = wine.classes, bgcol = bgcols[as.integer(xyfpredictions)],
     main = "another mapping plot")
## Show 'component planes'
set.seed(7)
sommap <- som(scale(wines), grid = somgrid(6, 4, "hexagonal"))
plot(sommap, type = "property", property = sommap$codes[,1],
     main = colnames(sommap$codes)[1])
## Another way to show clustering information
plot(sommap, type="dist.neighbours", main = "SOM neighbour distances")
## use hierarchical clustering to cluster the codebook vectors
som.hc <- cutree(hclust(dist(sommap$codes)), 5)
add.cluster.boundaries(sommap, som.hc)
## and the same for rectangular maps
set.seed(7)
sommap <- som(scale(wines),grid = somgrid(6, 4, "rectangular"))
plot(sommap, type="dist.neighbours", main = "SOM neighbour distances")
## use hierarchical clustering to cluster the codebook vectors
som.hc <- cutree(hclust(dist(sommap$codes)), 5)
add.cluster.boundaries(sommap, som.hc)

