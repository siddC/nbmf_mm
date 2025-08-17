# nbmf_mm/_rdeps.py
def install_r_requirements():
    """
    Install the R packages needed for nbmf-mm's R features.
    Requires: R installed on the system, and rpy2 (installed via the 'r' or 'all-r' extra).
    """
    try:
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects.vectors import StrVector
        utils = rpackages.importr("utils")
        # Choose a CRAN mirror; index 1 is typically the first in the list.
        utils.chooseCRANmirror(ind=1)
        needed = ["logisticPCA"]
        to_install = [pkg for pkg in needed if not rpackages.isinstalled(pkg)]
        if to_install:
            utils.install_packages(StrVector(to_install))
            print(f"Installed R packages: {', '.join(to_install)}")
        else:
            print("All required R packages already installed.")
    except Exception as e:
        print(
            "Failed to install R packages. Ensure R is installed and available, "
            "and that you installed 'nbmf-mm[all-r]' or 'nbmf-mm[r]'.\n"
            f"Details: {e}"
        )
