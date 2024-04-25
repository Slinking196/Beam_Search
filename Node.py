from CPMP.cpmp_ml import Layout

class Node():
    def __init__(self, f_lay: Layout, ac_lay: Layout):
        self.__f_lay = f_lay
        self.__ac_lay = ac_lay

    def get_f_lay(self) -> Layout:
        return self.__f_lay
    
    def get_ac_lay(self):
        return self.__ac_lay
    