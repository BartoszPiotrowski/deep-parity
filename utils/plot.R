library(ggplot2)
library(reshape)

args = commandArgs(trailingOnly=TRUE)
log <- args[1]
dir <- args[2]
log_lines <- readLines(log)
if (grepl('python', log_lines[1])){
	log_lines <- log_lines[-1]
}
train_lines <- log_lines[seq(1, length(log_lines), 2)]
valid_lines <- log_lines[seq(2, length(log_lines), 2)]
grep_numbers_from_string <- function(string){
	# and remove '.' from the end: 0.92198.
	as.double(sub('\\.$', '', grep('[0-9]', strsplit(string, ' ')[[1]], value=T)))
}
metrics_train <- do.call(rbind, lapply(train_lines, grep_numbers_from_string))
metrics_valid <- do.call(rbind, lapply(valid_lines, grep_numbers_from_string))
colnames(metrics_train) <- c('loss_train', 'accuracy_train')
colnames(metrics_valid) <- c('epoch', 'accuracy_valid')
metrics <- as.data.frame(cbind(metrics_valid, metrics_train))
loss_train <- data.frame(metrics$epoch, metrics$loss_train)
accuracy <- data.frame(metrics$epoch,
					   metrics$accuracy_train,
					   metrics$accuracy_valid)
colnames(loss_train) <- c('Epoch', 'Loss on training')
colnames(accuracy) <- c('Epoch', 'Accuracy on training', 'Accuracy on validation')

ggplot(loss_train, aes(x=`Epoch`, y=`Loss on training`)) + geom_line()
ggsave(paste(dir, sub('.log', '_loss.png', strsplit(log, '/')[[1]][-1], fixed=T),
			 sep='/'), device='png', width=12)

accuracy_melt <- melt(accuracy, c('Epoch'))
colnames(accuracy_melt) <- c('Epoch', 'Metric', 'Accuracy')
ggplot(accuracy_melt, aes(x=Epoch, y=Accuracy)) + geom_line(aes(color=Metric))
ggsave(paste(dir, sub('.log', '_accu.png', strsplit(log, '/')[[1]][-1], fixed=T),
			 sep='/'), device='png', width=12)
