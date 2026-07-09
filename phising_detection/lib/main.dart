import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import 'routes/app_router.dart';
import 'screens/phishing_scan/bloc/phishing_scan_bloc.dart';
import 'screens/phishing_scan/bloc/phishing_scan_event.dart';
import 'screens/phishing_scan/bloc/quiz_bloc.dart';
import 'theme/app_theme.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      systemNavigationBarColor: AppColors.background,
      systemNavigationBarIconBrightness: Brightness.light,
    ),
  );
  runApp(const PhishGuardApp());
}

class PhishGuardApp extends StatelessWidget {
  const PhishGuardApp({super.key});

  static final _appRouter = AppRouter();

  @override
  Widget build(BuildContext context) {
    return MultiBlocProvider(
      providers: [
        BlocProvider(
          create: (context) => QuizBloc(),
        ),
        BlocProvider(
          create: (context) => PhishingScanBloc()..add(PhishingScanBootstrap()),
        ),
      ],
      child: MaterialApp.router(
        title: 'PhishGuard',
        debugShowCheckedModeBanner: false,
        theme: AppTheme.dark,
        routerConfig: _appRouter.config(),
      ),
    );
  }
}
