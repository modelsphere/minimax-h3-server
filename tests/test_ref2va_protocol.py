from h3_service.config import REF2VA_TURBO_CONFIG_VERSION


def test_turbo_config_protocol_matches_the_deployment_guards():
    # The Helm chart (modelsphere/helm-charts, charts/minimax-h3/files/ref2va-config-check.py)
    # and h3_service/ref2va_profile.sh refuse a server image whose protocol is not 1.
    # Bumping it here means bumping both of those in the same release.
    assert REF2VA_TURBO_CONFIG_VERSION == 1
