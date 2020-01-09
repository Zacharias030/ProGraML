; ModuleID = '/scratch/talbn/classifyapp_code/validation/42/71.txt.cpp'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external global i8
@.str = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"%d \00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_71.txt.cpp, i8* null }]

define internal void @__cxx_global_var_init() #0 section ".text.startup" {
entry:
  call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* @_ZStL8__ioinit)
  %0 = call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i32 0, i32 0), i8* @__dso_handle) #1
  ret void
}

declare void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) #0

declare void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) #0

; Function Attrs: nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) #1

; Function Attrs: uwtable
define i32 @main() #2 {
entry:
  %retval = alloca i32, align 4
  %n = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  %d = alloca i32, align 4
  %e = alloca i32, align 4
  %f = alloca i32, align 4
  %g = alloca i32, align 4
  %h = alloca i32, align 4
  %saved_stack = alloca i8*
  store i32 0, i32* %retval
  %call = call i32 (i8*, ...) @scanf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i32* %n)
  %0 = load i32, i32* %n, align 4
  store i32 %0, i32* %h, align 4
  %1 = load i32, i32* %h, align 4
  %2 = zext i32 %1 to i64
  %3 = call i8* @llvm.stacksave()
  store i8* %3, i8** %saved_stack
  %vla = alloca i32, i64 %2, align 16
  store i32 0, i32* %b, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %4 = load i32, i32* %b, align 4
  %5 = load i32, i32* %n, align 4
  %sub = sub nsw i32 %5, 1
  %cmp = icmp sle i32 %4, %sub
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %6 = load i32, i32* %b, align 4
  %idxprom = sext i32 %6 to i64
  %arrayidx = getelementptr inbounds i32, i32* %vla, i64 %idxprom
  %call1 = call i32 (i8*, ...) @scanf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i32* %arrayidx)
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %7 = load i32, i32* %b, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, i32* %b, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %call2 = call i32 (i8*, ...) @scanf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i32* %c)
  store i32 0, i32* %f, align 4
  store i32 0, i32* %d, align 4
  br label %for.cond.3

for.cond.3:                                       ; preds = %for.inc.29, %for.end
  %8 = load i32, i32* %d, align 4
  %9 = load i32, i32* %n, align 4
  %sub4 = sub nsw i32 %9, 1
  %cmp5 = icmp sle i32 %8, %sub4
  br i1 %cmp5, label %for.body.6, label %for.end.31

for.body.6:                                       ; preds = %for.cond.3
  br label %while.cond

while.cond:                                       ; preds = %if.end, %for.body.6
  %10 = load i32, i32* %d, align 4
  %idxprom7 = sext i32 %10 to i64
  %arrayidx8 = getelementptr inbounds i32, i32* %vla, i64 %idxprom7
  %11 = load i32, i32* %arrayidx8, align 4
  %12 = load i32, i32* %c, align 4
  %cmp9 = icmp eq i32 %11, %12
  br i1 %cmp9, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %13 = load i32, i32* %d, align 4
  store i32 %13, i32* %e, align 4
  br label %for.cond.10

for.cond.10:                                      ; preds = %for.inc.18, %while.body
  %14 = load i32, i32* %e, align 4
  %15 = load i32, i32* %n, align 4
  %sub11 = sub nsw i32 %15, 2
  %cmp12 = icmp sle i32 %14, %sub11
  br i1 %cmp12, label %for.body.13, label %for.end.20

for.body.13:                                      ; preds = %for.cond.10
  %16 = load i32, i32* %e, align 4
  %add = add nsw i32 %16, 1
  %idxprom14 = sext i32 %add to i64
  %arrayidx15 = getelementptr inbounds i32, i32* %vla, i64 %idxprom14
  %17 = load i32, i32* %arrayidx15, align 4
  %18 = load i32, i32* %e, align 4
  %idxprom16 = sext i32 %18 to i64
  %arrayidx17 = getelementptr inbounds i32, i32* %vla, i64 %idxprom16
  store i32 %17, i32* %arrayidx17, align 4
  br label %for.inc.18

for.inc.18:                                       ; preds = %for.body.13
  %19 = load i32, i32* %e, align 4
  %inc19 = add nsw i32 %19, 1
  store i32 %inc19, i32* %e, align 4
  br label %for.cond.10

for.end.20:                                       ; preds = %for.cond.10
  %20 = load i32, i32* %c, align 4
  %cmp21 = icmp ne i32 %20, 0
  br i1 %cmp21, label %if.then, label %if.else

if.then:                                          ; preds = %for.end.20
  %21 = load i32, i32* %n, align 4
  %sub22 = sub nsw i32 %21, 1
  %idxprom23 = sext i32 %sub22 to i64
  %arrayidx24 = getelementptr inbounds i32, i32* %vla, i64 %idxprom23
  store i32 0, i32* %arrayidx24, align 4
  br label %if.end

if.else:                                          ; preds = %for.end.20
  %22 = load i32, i32* %n, align 4
  %sub25 = sub nsw i32 %22, 1
  %idxprom26 = sext i32 %sub25 to i64
  %arrayidx27 = getelementptr inbounds i32, i32* %vla, i64 %idxprom26
  store i32 1, i32* %arrayidx27, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %23 = load i32, i32* %f, align 4
  %add28 = add nsw i32 %23, 1
  store i32 %add28, i32* %f, align 4
  br label %while.cond

while.end:                                        ; preds = %while.cond
  br label %for.inc.29

for.inc.29:                                       ; preds = %while.end
  %24 = load i32, i32* %d, align 4
  %inc30 = add nsw i32 %24, 1
  store i32 %inc30, i32* %d, align 4
  br label %for.cond.3

for.end.31:                                       ; preds = %for.cond.3
  store i32 0, i32* %g, align 4
  br label %for.cond.32

for.cond.32:                                      ; preds = %for.inc.40, %for.end.31
  %25 = load i32, i32* %g, align 4
  %26 = load i32, i32* %n, align 4
  %27 = load i32, i32* %f, align 4
  %sub33 = sub nsw i32 %26, %27
  %sub34 = sub nsw i32 %sub33, 1
  %cmp35 = icmp slt i32 %25, %sub34
  br i1 %cmp35, label %for.body.36, label %for.end.42

for.body.36:                                      ; preds = %for.cond.32
  %28 = load i32, i32* %g, align 4
  %idxprom37 = sext i32 %28 to i64
  %arrayidx38 = getelementptr inbounds i32, i32* %vla, i64 %idxprom37
  %29 = load i32, i32* %arrayidx38, align 4
  %call39 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i32 0, i32 0), i32 %29)
  br label %for.inc.40

for.inc.40:                                       ; preds = %for.body.36
  %30 = load i32, i32* %g, align 4
  %inc41 = add nsw i32 %30, 1
  store i32 %inc41, i32* %g, align 4
  br label %for.cond.32

for.end.42:                                       ; preds = %for.cond.32
  %31 = load i32, i32* %g, align 4
  %idxprom43 = sext i32 %31 to i64
  %arrayidx44 = getelementptr inbounds i32, i32* %vla, i64 %idxprom43
  %32 = load i32, i32* %arrayidx44, align 4
  %call45 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i32 %32)
  store i32 0, i32* %retval
  %33 = load i8*, i8** %saved_stack
  call void @llvm.stackrestore(i8* %33)
  %34 = load i32, i32* %retval
  ret i32 %34
}

declare i32 @scanf(i8*, ...) #0

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #1

declare i32 @printf(i8*, ...) #0

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*) #1

define internal void @_GLOBAL__sub_I_71.txt.cpp() #0 section ".text.startup" {
entry:
  call void @__cxx_global_var_init()
  ret void
}

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.1 (tags/RELEASE_371/final)"}
