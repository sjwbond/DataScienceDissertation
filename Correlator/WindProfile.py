
from Profile import ProfileClass

class WindProfile(ProfileClass):

    def __init__(self, file_path, id, auto_blocks):

        ProfileClass.__init__(self, file_path, id)
        self.set_auto_correlation_blocks(auto_blocks)
        self.set_time_slice_by_month_and_period()

        self.calculate_distributions()

        self.make_analysis_matrix()
