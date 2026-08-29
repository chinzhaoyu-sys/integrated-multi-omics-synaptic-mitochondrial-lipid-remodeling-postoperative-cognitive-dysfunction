setwd('D:/爱谱蒂康/项目/HanJungang/workflow/data')
source('D:/爱谱蒂康/项目/func/DelSameRow.R')

# 合并表达矩阵 --------
# 分组信息
sampleInfo_sp <- read.csv('sampleInfo/sampleInfo_spatial.csv')
sampleInfo_su <- read.csv('sampleInfo/sampleInfo_serum.csv')

# 表达矩阵
mt_sp_pro <- read.csv('matrix/matrix_spatial_proteome.csv', row.names = 1)
mt_sp_pro$new_name <- sapply(rownames(mt_sp_pro), function(x) strsplit(x, '_')[[1]][2], simplify = T)
mt_sp_pro <- subset(mt_sp_pro, !is.na(new_name))
mt_sp_pro <- DelSameRow(data = mt_sp_pro, id_col = 'new_name', rm_col = 19)

mt_su_pro <- read.csv('matrix/matrix_serum_proteome.csv', row.names = 1)
mt_su_pro$new_name <- sapply(rownames(mt_su_pro), function(x) strsplit(x, '_')[[1]][2], simplify = T)
mt_su_pro <- subset(mt_su_pro, !is.na(new_name))
mt_su_pro <- DelSameRow(data = mt_su_pro, id_col = 'new_name', rm_col = 13)


mt_su_met <- read.csv('matrix/matrix_serum_metabolic.csv', row.names = 1)


# 保存数据
save(list = c('sampleInfo_sp','sampleInfo_su','mt_sp_pro','mt_su_pro', 'mt_su_met'),
     file = 'basic_data.RData')

# 合并差异表达结果 --------
diff_am <- read.csv('data/Diff/Diff_AM.csv')
diff_ca1 <- read.csv('data/Diff/Diff_CA1.csv')
diff_fra <- read.csv('data/Diff/Diff_FRA.csv')
save(list = c('diff_am', 'diff_ca1', 'diff_fra'), file = 'data/diff.RData')