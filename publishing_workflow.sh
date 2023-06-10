current_version="0.3.0"
bump_type="patch" # major / minor / patch


tag_name=v$current_version

####
# Build changelog
####


towncrier build
git commit -m "Building changelog"

####
# Bumping version number
####

bump2version --current-version $current_version $bump_type SBART/__init__.py setup.py --allow-dirty
git add pyproject.tom
git add SBART/__init__.py
git commit -m "new sBART version"

####
# Creating the tag
####

git tag -a v1.4 -m "New sBART release"

####
# Pushing the changes
####

git push --tags
####
# Finally, publish the package
####

poetry publish --build -u Kamuish
