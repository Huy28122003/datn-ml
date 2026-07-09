import 'package:auto_route/auto_route.dart';
import 'app_router.gr.dart'; // Generated file

@AutoRouterConfig(replaceInRouteName: 'Screen,Route')
class AppRouter extends $AppRouter {
  @override
  List<AutoRoute> get routes => [
        AutoRoute(page: SplashRoute.page, initial: true),
        AutoRoute(page: PhishingScanRoute.page),
      ];
}
