from semver import VersionInfo

__version__ = VersionInfo.parse("1.0.0-beta.4")

VERSION_STRING = f"v{__version__.major}.{__version__.minor}.{__version__.patch}-{__version__.prerelease}"
