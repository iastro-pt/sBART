from SBART.Base_Models.Template_Model import BaseTemplate


class LoadedTemplate(BaseTemplate):
    name = "Loaded Template"

    def __init__(self, path_to_file, logger):
        """
        Template object to hold information loaded from a .fits file, whilst maintaining the
        same interface as the other templates
        """
        super().__init__(0, logger=logger, loaded=True)
        self.load_from_file(path_to_file)
