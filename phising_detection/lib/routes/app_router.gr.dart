// GENERATED CODE - DO NOT MODIFY BY HAND

// **************************************************************************
// AutoRouterGenerator
// **************************************************************************

// ignore_for_file: type=lint
// coverage:ignore-file

// ignore_for_file: no_leading_underscores_for_library_prefixes
import 'package:auto_route/auto_route.dart' as _i4;
import 'package:phising_detection/screens/history/history_screen.dart' as _i1;
import 'package:phising_detection/screens/phishing_scan/phishing_scan_screen.dart'
    as _i2;
import 'package:phising_detection/screens/splash/splash_screen.dart' as _i3;

abstract class $AppRouter extends _i4.RootStackRouter {
  $AppRouter({super.navigatorKey});

  @override
  final Map<String, _i4.PageFactory> pagesMap = {
    HistoryRoute.name: (routeData) {
      return _i4.AutoRoutePage<dynamic>(
        routeData: routeData,
        child: const _i1.HistoryScreen(),
      );
    },
    PhishingScanRoute.name: (routeData) {
      return _i4.AutoRoutePage<dynamic>(
        routeData: routeData,
        child: const _i2.PhishingScanScreen(),
      );
    },
    SplashRoute.name: (routeData) {
      return _i4.AutoRoutePage<dynamic>(
        routeData: routeData,
        child: const _i3.SplashScreen(),
      );
    },
  };
}

/// generated route for
/// [_i1.HistoryScreen]
class HistoryRoute extends _i4.PageRouteInfo<void> {
  const HistoryRoute({List<_i4.PageRouteInfo>? children})
      : super(
          HistoryRoute.name,
          initialChildren: children,
        );

  static const String name = 'HistoryRoute';

  static const _i4.PageInfo<void> page = _i4.PageInfo<void>(name);
}

/// generated route for
/// [_i2.PhishingScanScreen]
class PhishingScanRoute extends _i4.PageRouteInfo<void> {
  const PhishingScanRoute({List<_i4.PageRouteInfo>? children})
      : super(
          PhishingScanRoute.name,
          initialChildren: children,
        );

  static const String name = 'PhishingScanRoute';

  static const _i4.PageInfo<void> page = _i4.PageInfo<void>(name);
}

/// generated route for
/// [_i3.SplashScreen]
class SplashRoute extends _i4.PageRouteInfo<void> {
  const SplashRoute({List<_i4.PageRouteInfo>? children})
      : super(
          SplashRoute.name,
          initialChildren: children,
        );

  static const String name = 'SplashRoute';

  static const _i4.PageInfo<void> page = _i4.PageInfo<void>(name);
}
