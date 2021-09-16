# -*- coding: utf-8 -*-
# @Time : 2019/8/9 17:28
# @Author : Edgar Qian
# @Email : qianneng_se@163.com
# @File : TripletFinder.py

from graphviz import Graph
from myProCon import MuiSite
from myProCon import Triplet


class TripletFinder:
    __resnum = 0
    __topnum = 0
    __ms = []
    __num_of_muincal = 0
    __tpsize = 0
    __num_of_triplets = 0
    __topos = 0
    __frompos = 0
    __pdiff = 0
    __pdiff_1 = 0
    __tps = []

    def __init__(self, mi):
        self.mi = mi
        self.__topnum = mi.get_top_num()
        print("topnum = {}".format(self.__topnum))
        self.__resnum = mi.get_residus_number()
        self.__num_of_muincal = mi.get_num_of_muincal()
        mut_obj = mi.get_mut_obj()

        for i in range(self.__resnum):
            self.__ms.append(MuiSite.MuiSite())

        size_of_site_j = []

        for i in range(self.__resnum):
            size_of_site_j.append(0)

        total_memory = 0

        for i in range(self.__topnum):
            size_of_site_j[int(mut_obj[i].get_site_1())] += 1
            total_memory += 1

        self.init_mui_site(size_of_site_j, total_memory)

        for i in range(self.__topnum):
            self.addms(mut_obj[i].get_site_1(), mut_obj[i].get_site_2())

        sj = 0
        for i in range(self.__resnum):
            sj = self.__ms[i].get_size_of_j()
            self.__ms[i].get_site_j().sort()

        self.__tpsize = int((((self.__resnum * self.__resnum * self.__resnum) - 3 *
                              (self.__resnum * self.__resnum) + 2 * self.__resnum) / 6 * 0.002))
        print("(allocated memory for number of triplets) tpsize = {}".format(self.__tpsize))

        self.__tps = []
        self.__num_of_triplets = 0
        key = 0
        e_site = 0
        col_1 = 0
        col_2 = 0
        keytmp = 0

        # indexes are i, j, k, l.
        # i, j pass through columns.
        # k, l pass through rows. in i column.
        # site1 is now implicitly used as the index i.
        for i in range(self.__resnum):
            # If there are no values in the rows of column ms[i] we do not have
            # triplets here.Continue to next column.
            if self.__ms[i].get_size_of_j() == 0:
                continue
            # j passes through all the columns on the right hand side of index i.
            for j in range(i + 1, self.__resnum):
                # k passes through the row elements of column i.
                for k in range(self.__ms[i].get_size_of_j()):
                    # If element i, k matches the index j and number of rows in
                    # column j are > 0 we will check for triplets.
                    if self.__ms[i].get_site_j()[k] == j and self.__ms[j].get_size_of_j() > 0:
                        # Starting to check the next element after i, k in column i.
                        # That is l = k + 1,  through each element in the rows of the i column.
                        for l in range(k + 1, self.__ms[i].get_size_of_j()):
                            # The key to check for is one of the elements below k. or l = k+1.
                            key = self.__ms[i].get_site_j()[l]
                            # Resetting the search area by setting topos to the
                            # number of elements in column j.
                            if keytmp != key:
                                self.__topos = self.__ms[j].get_size_of_j()
                                keytmp = key
                            # Resetting search area when the i column has changed.
                            if col_1 != i:
                                # New column, point the searched values to the
                                # first element in the new array to be searched.
                                col_1 = i
                                self.__frompos = 0
                                self.__topos = self.__ms[j].get_size_of_j()
                            # Resetting search area when the j column has changed.
                            if col_2 != j:
                                # New column, point the searched values to the
                                # first element in the new array to be searched.
                                col_2 = j
                                self.__frompos = 0
                                self.__topos = self.__ms[j].get_size_of_j()
                            self.__pdiff = self.__frompos  # Number of elements that has been searched through.
                            # Using exponential search to find the key or
                            # narrow search area for Binary search.
                            e_site = self.exp_search(self.__ms[j].get_site_j(), self.__pdiff,
                                                     self.__ms[j].get_size_of_j(), key)
                            self.__pdiff = self.__frompos  # Number of elements that has been searched through.
                            self.__pdiff = self.__topos  # Number of elements left to search for the key in.
                            if e_site <= 0:
                                if self.__pdiff <= 0:
                                    continue
                                else:
                                    e_site = self.bin_search(self.__ms[j].get_site_j(),
                                                             self.__pdiff, self.__pdiff_1, key)
                            # If esite > 0 then we have found the last element which is related to i, j.
                            # We have found a triplet.
                            if e_site > 0:
                                if self.__num_of_triplets >= self.__tpsize:
                                    self.__tpsize = self.__tpsize + int(
                                        (((self.__resnum * self.__resnum * self.__resnum)
                                          - 3 * (self.__resnum * self.__resnum) + 2 * self.__resnum)
                                         / 6 * (0.002 + (0.05 * mi.get_g_percent()))))
                                    print("tpzise2 = {}".format(self.__tpsize))
                                # Storing the triplet positions in the Triplet data structure.
                                tp = Triplet.Triplet()
                                tp.set_s1(i)
                                tp.set_s2(j)
                                tp.set_s3(e_site)
                                self.__tps.append(tp)
                                self.__num_of_triplets += 1

    def take_site_1(self, elem, i):
        return elem[i].get_site_1()

    def display_result(self):
        print("Attention!!!! numftriplets = {}".format(self.__num_of_triplets))

    # 注
    def init_mui_site(self, size_of_site_j, total_memory):
        counter = 0
        print("totalmemory = {}".format(total_memory))
        for i in range(self.__resnum):
            self.__ms[i].set_size_of_j(0)
            # print("ms[{}] = {}".format(i, self.__ms[i].get_size_of_j()))
            self.__ms[i].init_site_j(size_of_site_j[i])
            for j in range(counter, size_of_site_j[i]):
                self.__ms[i].set_site_j(j, 0)
            #    print("ms[{}].sitej[{}] = {}".format(i, j, self.__ms[i].get_site_j()[j]))
            counter += size_of_site_j[i]

    def addms(self, site_1, site_2):
        self.__ms[site_1].set_site_j(self.__ms[site_1].get_size_of_j(), site_2)
        self.__ms[site_1].set_size_of_j(self.__ms[site_1].get_size_of_j() + 1)

    def exp_search(self, sorted_array, first, last, key):
        """
        Search method that searches exponential valued indexes (0,1,2,4,8,16...)
        :param sorted_array: An array of sorted integers.
        :param first: The first position within the array to start the search from.
        :param last: The last position to end the search at.
        :param key: The value to search for within the array.
        :return:retval
        """
        retval = -1
        # j,l exponential indexes.
        j = 0  # The current exponential index.
        l = 0  # The previous exponential index.
        # jt, lt exponential indexes + the first position within an area to be searched.
        jt = 0  # The current exponential index + first,
        lt = 0  # lt=previous position we have searched through.
        for i in range(first, last - 1):
            # l,j index, lt,jt index+first
            l = j
            lt = j + first
            # Start the exponential incrementation when i > first + 1.
            # This means that we get the correct exponential incrementation of the  index.
            # if first=1, i=0 then j = j * 2, where j=0
            # if first=1, i=1 then j = j, where j=1
            # if first=1, i=2 then j = j * 2, where j=1 becoming j=2
            # if first=1, i=3 then j = j * 2, where j=2 becoming j=4
            if i != first + 1:
                j = j << 1
            # Setting jt to the actual search index, where first is constant within this scope,
            # and j incremented by 0, 1, 2, 4, 8.
            jt = j + first
            # Setting j=1 to start the exponential incrementation.
            if i == first:
                j = 1
            # Safety check, if the area to search for is invalid break immediately
            if first > last - 1:
                retval = -1
                break
            # If the search index has been incremented above the upper bound we
            # have finished the search and need to exit.
            if jt > last - 1:
                retval = -1
                break
            # If sortedArray[j] == key then we have found a key in the array
            # and will benefit from the fast search of expSearch.
            # Storing the next position where the value was found into frompos
            # for continous search of other values.
            # No information about the upper limit for next search, setting
            # topos to the length of the array.
            if sorted_array[jt] == key:
                retval = sorted_array[jt]
                self.__frompos = jt + 1
                self.__topos = last
                # frompos = sortedArray.length + jt + 1
                # topos = sortedArray.length + last
                break
            # If we have not found a match and the key is less than the current
            # array position this means we cannot find the value continuing the search.
            elif key < sorted_array[jt]:
                # if j is zero, then we will store the last index checked into
                # frompos for further search.
                # Now we should not search this array, using the current key,
                # anymore. Knowing that no value in this array may match the key.
                # To not search with binSearch, return an invalid search area
                # containing < 0 elements to search through.
                if jt == first:  # or j = 0
                    retval = -1
                    self.__frompos = first
                    self.__topos = last
                    # frompos = sortedArray.length + first
                    # topos = sortedArray.length + last
                    break
                # if lt > first means we will need to continue the search between an area 2^(i-1) and 2^i.
                # Or if the new search area does not contain any values
                if lt > first:
                    # Continue with binsearch between 2^(i-1) < key < 2^(i)
                    retval = -1
                    # If there are positions between 2^(i-1) and 2^i.
                    # Adding one position from last exponential index because we already checked this position
                    # Decreasing one position from current index position since we checked this one already.
                    if (jt - lt) > 1:
                        self.__frompos = lt + 1
                        self.__topos = jt
                        # frompos = sortedArray.length + lt+1
                        # topos = sortedArray.length + jt
                        break
                    # There are no elements between 2^(i-1) and (2^i) (lt and jt)
                    # which has not already been searched through or does not exist.
                    # Setting an invalid area to search for,
                    # which will notify that we do not have more elements to search for.
                    else:
                        self.__frompos = first
                        self.__topos = 0
                        # frompos = sortedArray.length + first
                        # topos = sortedArray.length
                        break
        return retval

    def bin_search(self, sorted_array, first, last, key):
        """
        Binary Search, searching for a key in a sorted array from position first to last.
        """
        retval = -1
        try:
            element = sorted_array.index(key)
        except:
            element = -1
        if element >= 0:
            # If a key is found, return the next position not searched, to continue the search from.
            self.__frompos = element + 1
            retval = sorted_array[element]
        return retval

    def get_triplets(self):
        return self.__tps

    def display_triplets(self):
        print("rank".ljust(7) + "site1".ljust(7) + "site2".ljust(7) + "site3".ljust(7))
        count = 1
        seq = self.mi.get_seq()
        for i in self.__tps:
            print("{}{}{}{}\n".format(str(count).ljust(7),
                                      (str(i.get_s1() + 1) + seq[0][i.get_s1()]).ljust(7),
                                      (str(i.get_s2() + 1) + seq[0][i.get_s2()]).ljust(7),
                                      (str(i.get_s3() + 1) + seq[0][i.get_s3()]).ljust(7)))
            count += 1

    def tps_to_file(self, file):
        fp = open(file, "w")
        fp.write("rank".ljust(7) + "site1".ljust(7) + "site2".ljust(7) + "site3".ljust(7) + "\n")
        count = 1
        seq = self.mi.get_seq()
        for i in self.__tps:
            fp.write("{}{}{}{}\n".format(str(count).ljust(7),
                                      (str(i.get_s1() + 1) + seq[0][i.get_s1()]).ljust(7),
                                      (str(i.get_s2() + 1) + seq[0][i.get_s2()]).ljust(7),
                                      (str(i.get_s3() + 1) + seq[0][i.get_s3()]).ljust(7)))
            count += 1
        fp.close()

    def show_graph(self, file):
        dot = Graph(comment="Triplets found among the covariant pairs", format="png")
        nodes = []
        edges = []
        tps = self.__tps
        for i in tps:
            node = i.get_s1()
            if node not in nodes:
                nodes.append(node)
            node = i.get_s2()
            if node not in nodes:
                nodes.append(node)
            node = i.get_s3()
            if node not in nodes:
                nodes.append(node)
        for i in nodes:
            dot.node(str(i), str(i), color="Blue")
        for i in tps:
            edge1 = [str(i.get_s1()), str(i.get_s2())]
            edge2 = [str(i.get_s1()), str(i.get_s3())]
            edge3 = [str(i.get_s2()), str(i.get_s3())]
            if edge1 not in edges:
                edges.append(edge1)
            if edge2 not in edges:
                edges.append(edge2)
            if edge3 not in edges:
                edges.append(edge3)
        for i in edges:
            dot.edge(i[0], i[1], color="red")
        dot.render(filename=file)
