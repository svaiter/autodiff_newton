## Differentiating IP solver
library(ggplot2)
library(reshape)
X <- read.delim("resDiffIP.csv", sep = ",", head = F)

colnames(X) <- c("nParam", "forwardMean", "forwardStd", "impDiffMean", "impDiffStd", "oneStepMean", "oneStepStd", "autodiffMean", "autodiffStd", "errOneStep", "errForward")


toPlot <- c()
diffMethods <- c("None", "Implicit", "One step", "Autodiff")
for (i in 1:length(diffMethods)){
	temp <- data.frame(nParam = X$nParam, time = X[,2*i], std = X[,2*i + 1], Gradient = diffMethods[i])
	toPlot <- rbind(toPlot, temp)
}

pl <- ggplot(toPlot) + 
	geom_line(aes(x = nParam, y = time, group = Gradient, colour = Gradient))  + 
	theme_bw()  + 
	geom_ribbon(aes(x=nParam, y=time, ymax=time +std, ymin=time -std, group = Gradient, colour = Gradient, fill = Gradient), alpha=0.2)
	
pdf("timingIPsolver.pdf", height = 3, width = 5)
print(pl)
dev.off()

toPlotIP <- toPlot
toPlotIP$pb <- "Interior point for QP"





## Differentiating Newton solver
library(ggplot2)
library(reshape)
X <- read.delim("resDiffNewton.csv", sep = ",", head = F)

colnames(X) <- c("nParam", "forwardMean", "forwardStd", "impDiffMean", "impDiffStd", "oneStepMean", "oneStepStd", "autodiffMean", "autodiffStd", "errOneStep", "errForward")


toPlot <- c()
diffMethods <- c("None", "Implicit", "One step", "Autodiff")
for (i in 1:length(diffMethods)){
	temp <- data.frame(nParam = X$nParam, time = X[,2*i], std = X[,2*i + 1], Gradient = diffMethods[i])
	toPlot <- rbind(toPlot, temp)
}

pl <- ggplot(toPlot) + 
	geom_line(aes(x = nParam, y = time, group = Gradient, colour = Gradient))  + 
	theme_bw()  + 
	geom_ribbon(aes(x=nParam, y=time, ymax=time +std, ymin=time -std, group = Gradient, colour = Gradient, fill = Gradient), alpha=0.2)
	
pdf("timingNewton.pdf", height = 3, width = 5)
print(pl)
dev.off()

toPlotNewton <- toPlot
toPlotNewton$pb <- "Newton for logistic regression"


toPlot <- rbind(toPlotIP,toPlotNewton)
toPlot$pb <- as.factor(toPlot$pb)

pl <- ggplot(toPlot) + 
	geom_line(aes(x = nParam, y = time, group = Gradient, colour = Gradient))  + 
	theme_bw()  + 
	geom_ribbon(aes(x=nParam, y=time, ymax=time +std, ymin=time -std, group = Gradient, colour = Gradient, fill = Gradient), alpha=0.2) + 
	facet_wrap(~pb, scales = "free")

pdf("timingSolvers.pdf", height = 3, width = 8)
print(pl)
dev.off()
