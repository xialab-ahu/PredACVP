inf = (r"./data/acvp-train.fasta")
outf = (r"./data/acvp_a.txt")
def readwrite(inf,outf):
    f = open(inf,'r')
    out = open(outf,'w')
    i = 1
    for line in f.readlines():
        lines = line.strip()
        print(lines)
        if i % 2 == 0:
            if i <423:
                for y in lines:
                    out.writelines(y)
                    out.writelines(' ')
                # out.writelines('.')
                out.writelines('\n')
            else:
                break
            i = i + 1
        else:
            i = i + 1
            continue
readwrite(inf,outf)


