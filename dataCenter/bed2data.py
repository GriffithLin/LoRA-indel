import pandas as pd
import numpy as np



def before_mutation(seq, crop_len, seq_len):
    seq = seq.upper()
    out_len = crop_len*2
    result = seq[ seq_len+1-crop_len:]
    return result[:out_len]


# 都从中间的下一位开始插入删除
def mutation(del_seq, ins_seq, seq, crop_len, seq_len):
    seq = seq.upper()
    out_len = crop_len * 2

    del_seq = str(del_seq)
    ins_seq = str(ins_seq)
    assert not (len(del_seq) > 1 & len(ins_seq) > 1)
    #     if Type == "Indel":
    # #         不找依据
    #         result = seq[seq_len-crop_len:seq_len] + ins_seq + seq[seq_len + len(del_seq):]
    #         return result[:out_len]

    result = seq[seq_len - crop_len + 1: seq_len + 1] + ins_seq[1:] + seq[seq_len + len(del_seq):]
    return result[:out_len]


def before_mutation_context(del_seq, ins_seq, seq, crop_len, seq_len):
    seq = seq.upper()
    out_len = crop_len * 2
    result = seq[seq_len + 1 - crop_len:]
    if len(del_seq) < len(ins_seq) :
        return result[:out_len]
    else:
        del_len = len(del_seq) - len(ins_seq)
        return result[:out_len + del_len]

# 都从中间的下一位开始插入删除
def mutation_context(del_seq, ins_seq, seq, crop_len, seq_len):
    seq = seq.upper()
    out_len = crop_len * 2

    assert not (len(del_seq) > 1 & len(ins_seq) > 1)
    #     if Type == "Indel":
    # #         不找依据
    #         result = seq[seq_len-crop_len:seq_len] + ins_seq + seq[seq_len + len(del_seq):]
    #         return result[:out_len]

    result = seq[seq_len - crop_len + 1: seq_len + 1] + ins_seq[1:] + seq[seq_len + len(del_seq):]
    if len(del_seq) < len(ins_seq):
        add_len = len(ins_seq) - len(del_seq)
        return result[:out_len + add_len]
    else:
        return result[:out_len]

def mutation_type(ref, alt):
    ref = str(ref)
    alt = str(alt)
    if abs(len(ref)-len(alt))%3==0:
        return "inframe"
    return "frameshift"

# vcf_file_path_list = ["train.vcf", "test.vcf", "DDD.vcf"]
# src = "gnomAD"
vcf_file_path_list = ["DA_target.vcf", "DA_target_test.vcf", "source_Data.vcf", "source_train.vcf"]


bed_len = 300
context = False
# context = True
crop_len_list = [50, 70, 100, 150, 200, 225]

if __name__ == "__main__":
    for crop_len in crop_len_list:
        for vcf_file_path in vcf_file_path_list:
            if context:
                bed_out = pd.read_csv(vcf_file_path[:-4] + "_" + str(bed_len) + ".out", sep='\t')
                bed_out = bed_out.iloc[lambda x: x.index % 2 == 0]
                bed_out = bed_out.reset_index(drop=True)
                bed_out.columns = ["seq"]

                vcf = pd.read_csv(vcf_file_path, sep='\t', header=0)
                vcf = vcf.head(len(bed_out))

                bed_out["src"] = vcf["INFO"].apply(lambda x: "hgmd" if x.split("_")[-1] == "1" else "gnomAD" )
                bed_out["REF"] = vcf["REF"].astype("str")
                bed_out["ALT"] = vcf["ALT"].astype("str")
                bed_out["CHROM"] = vcf["#CHROM"]
                bed_out["ID"] = vcf["ID"]
                bed_out["POS"] = vcf["POS"]
                bed_out["HGVS_cdna"] = vcf["INFO"]
                bed_out['before_mutation'] = bed_out.apply(lambda x: before_mutation_context(x['REF'], x['ALT'],x['seq'], crop_len, bed_len), axis=1)
                bed_out['after_mutation'] = bed_out.apply(lambda x: mutation_context(x['REF'], x['ALT'], x['seq'], crop_len, bed_len),
                                                        axis=1)
                print(len(bed_out))
                bed_out.drop_duplicates(subset = ["REF", "ALT", "CHROM", "POS"], keep="first", inplace = True)
                print(len(bed_out))

                bed_out["mutation_type"] = bed_out.apply(lambda x: mutation_type(x["REF"], x["ALT"]) , axis = 1)


                inframe_out = bed_out[bed_out["mutation_type"] == "inframe"]
                print(len(inframe_out))
                save_inframe = inframe_out[["REF", 'ALT', 'before_mutation', 'after_mutation', "CHROM", "HGVS_cdna", "POS", "src"]]
                # print(save_data)
                save_inframe.to_csv( vcf_file_path[:-4] + str(crop_len) + "_context.csv")

                # frameshift_out = bed_out[bed_out["mutation_type"] == "frameshift"]
                # save_frameshift = frameshift_out[["REF", 'ALT', 'before_mutation', 'after_mutation', "CHROM", "HGVS_cdna", "POS"]]
                # # print(save_data)
                # save_frameshift.to_csv("frameshift_" + vcf_file_path[:-4] + str(crop_len) + "_seq.csv")
            else:
                bed_out = pd.read_csv(vcf_file_path[:-4] + "_" + str(bed_len) + ".out", sep='\t')
                bed_out = bed_out.iloc[lambda x: x.index % 2 == 0]
                bed_out = bed_out.reset_index(drop=True)
                bed_out.columns = ["seq"]

                vcf = pd.read_csv(vcf_file_path, sep='\t', header=0)
                vcf = vcf.head(len(bed_out))

                bed_out["src"] = vcf["INFO"].apply(lambda x: "hgmd" if x.split("_")[-1] == "1" else "gnomAD" )
                bed_out["REF"] = vcf["REF"].astype("str")
                bed_out["ALT"] = vcf["ALT"].astype("str")
                bed_out["CHROM"] = vcf["#CHROM"]
                bed_out["ID"] = vcf["ID"]
                bed_out["POS"] = vcf["POS"]
                bed_out["HGVS_cdna"] = vcf["INFO"]
                bed_out['before_mutation'] = bed_out.apply(lambda x: before_mutation(x['seq'], crop_len, bed_len), axis=1)
                bed_out['after_mutation'] = bed_out.apply(lambda x: mutation(x['REF'], x['ALT'], x['seq'], crop_len, bed_len),
                                                        axis=1)
                print(len(bed_out))
                bed_out.drop_duplicates(subset =  ["REF", "ALT", "CHROM", "POS"], keep="first", inplace = True)

                print(len(bed_out))

                bed_out["mutation_type"] = bed_out.apply(lambda x: mutation_type(x["REF"], x["ALT"]) , axis = 1)


                inframe_out = bed_out[bed_out["mutation_type"] == "inframe"]
                print(len(bed_out))

                save_inframe = inframe_out[["REF", 'ALT', 'before_mutation', 'after_mutation', "CHROM", "HGVS_cdna", "POS", "src"]]
                # print(save_data)
                save_inframe.to_csv(vcf_file_path[:-4] + str(crop_len) + ".csv")

                # frameshift_out = bed_out[bed_out["mutation_type"] == "frameshift"]
                # save_frameshift = frameshift_out[["REF", 'ALT', 'before_mutation', 'after_mutation', "CHROM", "HGVS_cdna", "POS"]]
                # # print(save_data)
                # save_frameshift.to_csv("frameshift_" + vcf_file_path[:-4] + str(crop_len) + "_seq.csv")

